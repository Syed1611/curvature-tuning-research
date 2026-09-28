"""
Formula-SWCT: direct stage-wise beta prediction followed by one downstream run.

Final research pipeline
-----------------------
FIT (done once during method development):
    Frozen ImageNet ResNet-18 + calibration datasets
        -> collect per-stage statistics
        -> fit a small ridge formula to previously learned SW-CT betas
        -> save formula coefficients

RUN (for a target dataset):
    Frozen ImageNet ResNet-18 + target training split
        -> collect the same per-stage statistics
        -> predict beta_1..beta_4 directly (no beta sweep, no beta training)
        -> freeze predicted betas
        -> train ONE linear probe
        -> evaluate once on the test set

The predictor never uses the target test set, never starts from beta=0.78,
and never performs candidate-by-candidate beta training.
"""

import argparse
import copy
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from utils.data import get_data_loaders, DATASET_TO_NUM_CLASSES
from utils.utils import get_pretrained_model, fix_seed
from utils.curvature_tuning import (
    replace_resnet_relu_stagewise,
    get_stage_betas,
)
from train import WarmUpLR


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

BETA_MIN = 0.70
BETA_MAX = 0.99

# Fixed-beta calibration targets selected from stage-isolated
# validation-accuracy landscapes.
#
# These targets match the final deployment setting:
# beta values are predicted BEFORE downstream classifier training
# and remain fixed during the single downstream training run.
#
# Selection rule:
# choose the smallest beta achieving the maximum validation accuracy
# among the stage-isolated probe values.
CALIBRATION_TARGETS = {
    "beans": [0.85, 0.85, 0.85, 0.80],
    "dtd": [0.95, 0.95, 0.99, 0.99],
    "flowers102": [0.99, 0.95, 0.99, 0.90],
}

CALIBRATION_SEEDS = [42, 43, 44]
FEATURE_NAMES = [
    "log_variance",
    "sparsity",
    "log_mean_abs",
    "log_class_separation",
    "depth",
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Formula-SWCT: direct beta prediction + one training run"
    )

    parser.add_argument(
        "--mode",
        choices=["fit", "run"],
        required=True,
        help="fit the formula once, or run one target dataset with the saved formula",
    )
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--pretrained_ds", type=str, default="imagenet")
    parser.add_argument("--transfer_ds", type=str, default="imagenette")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_bs", type=int, default=32)
    parser.add_argument("--test_bs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--stat_batches",
        type=int,
        default=0,
        help="0 uses the entire training loader when collecting frozen statistics",
    )
    parser.add_argument(
        "--ridge_lambda",
        type=float,
        default=0.10,
        help="fixed L2 regularization for the small formula fit",
    )
    parser.add_argument(
        "--formula_path",
        type=str,
        default="./results/formula_swct_model.json",
    )

    return parser.parse_args()


def _class_separation(features, labels):
    """Scale-invariant between-class / within-class separation score."""
    features = features.double()
    labels = labels.long()

    global_mean = features.mean(dim=0)
    between_sum = torch.tensor(0.0, dtype=torch.float64)
    within_sum = torch.tensor(0.0, dtype=torch.float64)
    total = features.shape[0]

    for class_id in torch.unique(labels):
        mask = labels == class_id
        class_features = features[mask]
        class_mean = class_features.mean(dim=0)
        n = class_features.shape[0]

        between_sum += n * torch.mean((class_mean - global_mean) ** 2)
        within_sum += torch.sum(
            torch.mean((class_features - class_mean) ** 2, dim=1)
        )

    between = between_sum / max(total, 1)
    within = within_sum / max(total, 1)
    ratio = between / (within + 1e-12)
    return float(torch.log1p(ratio).item())


def collect_stage_statistics(model, loader, seed, max_batches=0):
    """
    Collect five pre-training descriptors for ResNet stages 1..4.

    Hooks are placed on layer1..layer4 outputs of the original pretrained
    ReLU network. No classifier or beta is trained.
    """
    if not all(hasattr(model, name) for name in ["layer1", "layer2", "layer3", "layer4"]):
        raise ValueError("Formula-SWCT currently supports ResNet-style layer1..layer4 models only.")

    fix_seed(seed)
    model.eval()

    stage_modules = [model.layer1, model.layer2, model.layer3, model.layer4]
    batch_outputs = [None, None, None, None]
    handles = []

    def make_hook(index):
        def hook(_module, _inputs, output):
            batch_outputs[index] = output.detach()
        return hook

    for index, module in enumerate(stage_modules):
        handles.append(module.register_forward_hook(make_hook(index)))

    sums = [0.0] * 4
    sq_sums = [0.0] * 4
    abs_sums = [0.0] * 4
    zero_counts = [0] * 4
    counts = [0] * 4
    pooled_features = [[] for _ in range(4)]
    all_labels = []

    try:
        with torch.inference_mode():
            for batch_index, (inputs, labels) in enumerate(loader):
                if max_batches > 0 and batch_index >= max_batches:
                    break

                inputs = inputs.to(device)
                batch_outputs[:] = [None, None, None, None]
                _ = model(inputs)
                all_labels.append(labels.cpu())

                for stage_index, output in enumerate(batch_outputs):
                    if output is None:
                        raise RuntimeError(f"Stage {stage_index + 1} hook did not fire.")

                    values = output.float()
                    sums[stage_index] += values.sum().item()
                    sq_sums[stage_index] += (values * values).sum().item()
                    abs_sums[stage_index] += values.abs().sum().item()
                    zero_counts[stage_index] += (values.abs() <= 1e-8).sum().item()
                    counts[stage_index] += values.numel()

                    pooled = F.adaptive_avg_pool2d(values, 1).flatten(1)
                    pooled_features[stage_index].append(pooled.cpu())
    finally:
        for handle in handles:
            handle.remove()

    labels = torch.cat(all_labels)
    stats = []

    for stage_index in range(4):
        n = counts[stage_index]
        mean = sums[stage_index] / n
        mean_sq = sq_sums[stage_index] / n
        variance = max(mean_sq - mean * mean, 1e-12)
        mean_abs = max(abs_sums[stage_index] / n, 1e-12)
        sparsity = zero_counts[stage_index] / n

        features = torch.cat(pooled_features[stage_index], dim=0)
        separation = _class_separation(features, labels)
        depth = stage_index / 3.0

        feature_vector = [
            math.log(variance + 1e-12),
            float(sparsity),
            math.log(mean_abs + 1e-12),
            float(separation),
            float(depth),
        ]

        stats.append({
            "stage": stage_index + 1,
            "feature_vector": feature_vector,
            "features": dict(zip(FEATURE_NAMES, feature_vector)),
        })

    return stats


def fit_ridge_formula(X, y, ridge_lambda):
    """Closed-form ridge regression with an unpenalized intercept."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0
    Z = (X - feature_mean) / feature_std

    design = np.concatenate([np.ones((Z.shape[0], 1)), Z], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_lambda
    penalty[0, 0] = 0.0

    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ y,
    )

    predictions = np.clip(design @ coefficients, BETA_MIN, BETA_MAX)
    mae = float(np.mean(np.abs(predictions - y)))

    return {
        "intercept": float(coefficients[0]),
        "weights": coefficients[1:].tolist(),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "ridge_lambda": float(ridge_lambda),
        "training_mae": mae,
    }


def predict_from_formula(stage_stats, formula):
    mean = np.asarray(formula["feature_mean"], dtype=np.float64)
    std = np.asarray(formula["feature_std"], dtype=np.float64)
    weights = np.asarray(formula["weights"], dtype=np.float64)
    intercept = float(formula["intercept"])

    predictions = []
    for stage in stage_stats:
        x = np.asarray(stage["feature_vector"], dtype=np.float64)
        z = (x - mean) / std
        beta = intercept + float(z @ weights)
        beta = float(np.clip(beta, BETA_MIN, BETA_MAX))
        predictions.append(beta)

    return predictions


def set_stage_betas(model, betas):
    with torch.no_grad():
        for raw_param, beta in zip(model.stage_raw_betas.parameters(), betas):
            beta_tensor = torch.tensor(
                beta,
                dtype=raw_param.dtype,
                device=raw_param.device,
            )
            raw_param.copy_(torch.logit(beta_tensor))

    for param in model.stage_raw_betas.parameters():
        param.requires_grad = False


def extract_fixed_features(model, loader):
    model.eval()
    features = []
    labels = []

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = torch.flatten(outputs, 1)
            features.append(outputs.cpu())
            labels.append(targets.cpu())

    return torch.cat(features), torch.cat(labels)


def evaluate_classifier(classifier, loader, criterion):
    classifier.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = classifier(features)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            loss_sum += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += batch_size

    return loss_sum / total, 100.0 * correct / total


def train_one_fixed_beta_probe(
    stage_model,
    train_loader,
    val_loader,
    test_loader,
    num_classes,
    train_bs,
    eval_bs,
    epochs,
    seed,
):
    """One S-CT-style linear probe after betas have already been predicted."""
    for param in stage_model.parameters():
        param.requires_grad = False

    feature_model = copy.deepcopy(stage_model)
    feature_model.fc = nn.Identity()
    feature_model = feature_model.to(device)

    print("\nExtracting fixed Formula-SWCT features once...")
    train_features, train_labels = extract_fixed_features(feature_model, train_loader)
    val_features, val_labels = extract_fixed_features(feature_model, val_loader)
    test_features, test_labels = extract_fixed_features(feature_model, test_loader)

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    fix_seed(seed)
    train_feature_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=2,
    )
    val_feature_loader = DataLoader(
        val_dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=2,
    )
    test_feature_loader = DataLoader(
        test_dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=2,
    )

    classifier = nn.Linear(train_features.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    warmup_scheduler = WarmUpLR(optimizer, len(train_feature_loader))
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 20],
        gamma=0.1,
    )

    best_state = None
    best_val_acc = -1.0
    best_epoch = None

    print("Training ONE downstream classifier...")
    for epoch in range(1, epochs + 1):
        classifier.train()

        for features, targets in train_feature_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = classifier(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if epoch <= 1:
                warmup_scheduler.step()

        val_loss, val_acc = evaluate_classifier(
            classifier,
            val_feature_loader,
            criterion,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"val_loss={val_loss:.6f} | "
            f"val_acc={val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(classifier.state_dict())

        scheduler.step()

    classifier.load_state_dict(best_state)
    test_loss, test_acc = evaluate_classifier(
        classifier,
        test_feature_loader,
        criterion,
    )

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def fit_mode(args):
    if args.model != "resnet18":
        raise ValueError("The current Formula-SWCT calibration is for ResNet-18 only.")

    X = []
    y = []
    rows = []

    print("=" * 82)
    print("FORMULA-SWCT: FIT DIRECT BETA FORMULA")
    print("=" * 82)
    print("No beta sweep and no downstream training are performed here.")

    for dataset_name, targets in CALIBRATION_TARGETS.items():
        for seed in CALIBRATION_SEEDS:
            print(f"\nCollecting frozen statistics: {dataset_name}, seed {seed}")

            dataset = f"{args.pretrained_ds}_to_{dataset_name}"
            train_loader, _test_loader, _val_loader = get_data_loaders(
                dataset,
                seed=seed,
                train_batch_size=args.train_bs,
                test_batch_size=args.test_bs,
            )

            model = get_pretrained_model(args.pretrained_ds, args.model).to(device)
            model.fc = nn.Identity()
            for param in model.parameters():
                param.requires_grad = False

            stats = collect_stage_statistics(
                model,
                train_loader,
                seed=seed,
                max_batches=args.stat_batches,
            )

            for stage_index, stage in enumerate(stats):
                target_beta = float(np.clip(targets[stage_index], BETA_MIN, BETA_MAX))
                X.append(stage["feature_vector"])
                y.append(target_beta)
                rows.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "stage": stage_index + 1,
                    "features": stage["features"],
                    "target_beta_original": targets[stage_index],
                    "target_beta_used": target_beta,
                })

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    formula = fit_ridge_formula(X, y, args.ridge_lambda)

    output = {
        "method": "Formula-SWCT",
        "architecture": args.model,
        "pretrained_dataset": args.pretrained_ds,
        "beta_min": BETA_MIN,
        "beta_max": BETA_MAX,
        "feature_names": FEATURE_NAMES,
        "calibration_targets": CALIBRATION_TARGETS,
        "calibration_seeds": CALIBRATION_SEEDS,
        "formula": formula,
        "calibration_rows": rows,
    }

    os.makedirs(os.path.dirname(args.formula_path) or ".", exist_ok=True)
    with open(args.formula_path, "w") as handle:
        json.dump(output, handle, indent=4)

    print("\n" + "=" * 82)
    print("FITTED FORMULA")
    print("=" * 82)
    print(f"beta_hat = clip(b0 + sum(w_j * z_j), {BETA_MIN}, {BETA_MAX})")
    print(f"intercept: {formula['intercept']:.8f}")
    for name, weight in zip(FEATURE_NAMES, formula["weights"]):
        print(f"{name:>24}: {weight:+.8f}")
    print(f"Calibration MAE: {formula['training_mae']:.6f}")
    print(f"Saved formula: {args.formula_path}")


def run_mode(args):
    if not os.path.exists(args.formula_path):
        raise FileNotFoundError(
            f"Formula file not found: {args.formula_path}. Run --mode fit first."
        )

    with open(args.formula_path, "r") as handle:
        saved = json.load(handle)

    formula = saved["formula"]
    dataset = f"{args.pretrained_ds}_to_{args.transfer_ds}"

    print("=" * 82)
    print("FORMULA-SWCT: PREDICT FIRST, TRAIN ONCE")
    print("=" * 82)
    print("Dataset:", args.transfer_ds)
    print("Seed:", args.seed)
    print("No beta sweep. No beta optimization. No 0.78 anchor.")

    train_loader, test_loader, val_loader = get_data_loaders(
        dataset,
        seed=args.seed,
        train_batch_size=args.train_bs,
        test_batch_size=args.test_bs,
    )

    # 1) Predict beta BEFORE downstream training.
    predictor_model = get_pretrained_model(args.pretrained_ds, args.model).to(device)
    predictor_model.fc = nn.Identity()
    for param in predictor_model.parameters():
        param.requires_grad = False

    stage_stats = collect_stage_statistics(
        predictor_model,
        train_loader,
        seed=args.seed,
        max_batches=args.stat_batches,
    )
    predicted_betas = predict_from_formula(stage_stats, formula)

    print("\nPredicted betas BEFORE training:")
    for index, beta in enumerate(predicted_betas, start=1):
        print(f"Stage {index}: {beta:.6f}")
    print("Predicted beta vector:", [round(x, 6) for x in predicted_betas])

    # 2) Build the fixed-beta model only after prediction is complete.
    base_model = get_pretrained_model(args.pretrained_ds, args.model)
    for param in base_model.parameters():
        param.requires_grad = False

    base_model.fc = nn.Linear(
        base_model.fc.in_features,
        DATASET_TO_NUM_CLASSES[args.transfer_ds],
    )
    base_model = base_model.to(device)

    stage_model = replace_resnet_relu_stagewise(
        copy.deepcopy(base_model),
        init_beta=predicted_betas[0],  # construction value comes from the formula itself
        coeff=0.5,
    ).to(device)
    set_stage_betas(stage_model, predicted_betas)

    loaded_betas = get_stage_betas(stage_model)
    print("Loaded fixed betas:", [round(x, 6) for x in loaded_betas])

    # 3) Exactly one downstream linear-probe training run.
    result = train_one_fixed_beta_probe(
        stage_model,
        train_loader,
        val_loader,
        test_loader,
        DATASET_TO_NUM_CLASSES[args.transfer_ds],
        args.train_bs,
        args.test_bs,
        args.epochs,
        args.seed,
    )

    print("\n" + "=" * 82)
    print("FORMULA-SWCT FINAL RESULT")
    print("=" * 82)
    print("Predicted betas:", [round(x, 6) for x in predicted_betas])
    print(f"Best validation accuracy: {result['best_val_acc']:.2f}%")
    print(f"Best validation epoch: {result['best_epoch']}")
    print(f"Test loss: {result['test_loss']:.6f}")
    print(f"Test accuracy: {result['test_acc']:.2f}%")

    os.makedirs("./results", exist_ok=True)
    safe_transfer_ds = args.transfer_ds.replace("/", "-")

    result_path = (
        f"./results/formula_swct_{args.pretrained_ds}_to_"
        f"{safe_transfer_ds}_{args.model}_seed{args.seed}.json"
    )
    with open(result_path, "w") as handle:
        json.dump({
            "dataset": args.transfer_ds,
            "seed": args.seed,
            "predicted_betas": predicted_betas,
            "stage_statistics": stage_stats,
            "epochs": args.epochs,
            **result,
        }, handle, indent=4)

    print("Saved result:", result_path)


def main():
    args = get_args()
    fix_seed(args.seed)

    if args.mode == "fit":
        fit_mode(args)
    else:
        run_mode(args)


if __name__ == "__main__":
    main()
