"""
Fixed-Classifier Beta Landscape Probe.

Goal:
    Estimate stage-wise beta behavior over [0.70, 0.99]
    using ONE fixed downstream classifier.

Protocol:
1. Load pretrained ImageNet ResNet-18 with normal ReLUs.
2. Freeze backbone.
3. Train ONE baseline linear classifier on ReLU features.
4. Freeze that classifier permanently.
5. For beta in:
       0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99
   replace ReLUs with SW-CT units.
6. Set all four stage betas to the same probe beta.
7. DO NOT retrain the classifier.
8. Measure dL/dbeta for each stage using the same classifier.
9. Find gradient zero crossings.

No test set.
No learned SW-CT beta targets.
No 0.78 anchor.
"""

import argparse
import copy
import json
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from utils.data import get_data_loaders

from utils.utils import (
    get_pretrained_model,
    fix_seed,
)

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


PROBE_BETAS = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
]


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
    )

    parser.add_argument(
        "--pretrained_ds",
        type=str,
        default="imagenet",
    )

    parser.add_argument(
        "--transfer_ds",
        type=str,
        default="beans",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--train_bs",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--test_bs",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--classifier_epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--num_gradient_batches",
        type=int,
        default=0,
        help="0 means use the full training loader.",
    )

    return parser.parse_args()


@torch.inference_mode()
def extract_features(
    model,
    loader,
):

    model.eval()

    features = []
    labels = []

    for inputs, targets in loader:

        inputs = inputs.to(device)

        outputs = model(inputs)

        outputs = torch.flatten(
            outputs,
            1,
        )

        features.append(
            outputs.cpu()
        )

        labels.append(
            targets.cpu()
        )

    return (
        torch.cat(features),
        torch.cat(labels),
    )


def evaluate_classifier(
    classifier,
    loader,
    criterion,
):

    classifier.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for features, targets in loader:

            features = features.to(device)
            targets = targets.to(device)

            outputs = classifier(
                features
            )

            loss = criterion(
                outputs,
                targets,
            )

            batch_size = targets.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions
                == targets
            ).sum().item()

            total += batch_size

    return (
        total_loss / total,
        100.0 * correct / total,
    )


@torch.inference_mode()
def evaluate_full_model(
    model,
    loader,
):

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets,
        )

        batch_size = targets.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions
            == targets
        ).sum().item()

        total += batch_size

    return (
        total_loss / total,
        100.0 * correct / total,
    )


def train_baseline_classifier(
    model,
    train_loader,
    val_loader,
    train_bs,
    val_bs,
    epochs,
    seed,
):
    """
    Train ONE classifier on original ReLU features.

    This classifier will later be frozen and reused
    for every CT beta probe.
    """

    for param in model.parameters():
        param.requires_grad = False

    feature_dim = model.fc.in_features

    # Remove ImageNet classifier.
    model.fc = nn.Identity()

    model.eval()

    print(
        "\nExtracting ORIGINAL RELU train features..."
    )

    fix_seed(seed)

    train_features, train_labels = (
        extract_features(
            model,
            train_loader,
        )
    )

    print(
        "Extracting ORIGINAL RELU validation features..."
    )

    fix_seed(seed)

    val_features, val_labels = (
        extract_features(
            model,
            val_loader,
        )
    )

    print(
        "Train feature shape:",
        tuple(train_features.shape),
    )

    train_dataset = TensorDataset(
        train_features,
        train_labels,
    )

    val_dataset = TensorDataset(
        val_features,
        val_labels,
    )

    fix_seed(seed)

    train_feature_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=2,
    )

    val_feature_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=2,
    )

    num_classes = (
        int(train_labels.max().item())
        + 1
    )

    classifier = nn.Linear(
        feature_dim,
        num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        classifier.parameters(),
        lr=1e-3,
    )

    warmup_scheduler = WarmUpLR(
        optimizer,
        len(train_feature_loader),
    )

    scheduler = (
        optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[10, 20],
            gamma=0.1,
        )
    )

    best_state = None
    best_val_acc = -1.0
    best_epoch = None

    print(
        "\nTraining ONE baseline ReLU classifier..."
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        classifier.train()

        for features, targets in (
            train_feature_loader
        ):

            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = classifier(
                features
            )

            loss = criterion(
                outputs,
                targets,
            )

            loss.backward()

            optimizer.step()

            if epoch <= 1:
                warmup_scheduler.step()

        val_loss, val_acc = (
            evaluate_classifier(
                classifier,
                val_feature_loader,
                criterion,
            )
        )

        print(
            f"Epoch {epoch:02d} | "
            f"val_loss={val_loss:.6f} | "
            f"val_acc={val_acc:.2f}%"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            best_epoch = epoch

            best_state = copy.deepcopy(
                classifier.state_dict()
            )

        scheduler.step()

    classifier.load_state_dict(
        best_state
    )

    # Attach the ONE trained classifier.
    model.fc = classifier

    # Freeze it permanently.
    for param in model.fc.parameters():
        param.requires_grad = False

    print(
        f"\nBaseline classifier selected at "
        f"epoch {best_epoch}"
    )

    print(
        f"Baseline ReLU validation accuracy: "
        f"{best_val_acc:.2f}%"
    )

    return (
        model,
        best_val_acc,
        best_epoch,
    )


def measure_beta_gradients(
    model,
    train_loader,
    beta_value,
    seed,
    num_batches=0,
):

    # Freeze everything.
    for param in model.parameters():
        param.requires_grad = False

    beta_params = list(
        model.stage_raw_betas.parameters()
    )

    # Enable ONLY beta gradients.
    for param in beta_params:
        param.requires_grad = True

    model.eval()

    criterion = nn.CrossEntropyLoss()

    raw_gradient_sum = torch.zeros(
        4,
        dtype=torch.float64,
        device=device,
    )

    total_examples = 0
    total_loss = 0.0
    batches_used = 0

    # Same augmentation/random sequence at each beta.
    fix_seed(seed)

    for batch_idx, (
        inputs,
        targets,
    ) in enumerate(train_loader):

        if (
            num_batches > 0
            and batch_idx >= num_batches
        ):
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets,
        )

        gradients = torch.autograd.grad(
            loss,
            beta_params,
            create_graph=False,
            retain_graph=False,
        )

        gradient_vector = torch.stack(
            gradients
        ).double()

        batch_size = targets.size(0)

        raw_gradient_sum += (
            gradient_vector.detach()
            * batch_size
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_examples += batch_size
        batches_used += 1

    mean_raw_gradient = (
        raw_gradient_sum
        / total_examples
    )

    # beta = sigmoid(raw_beta)
    #
    # dL/draw =
    # dL/dbeta * beta*(1-beta)
    sigmoid_scale = (
        beta_value
        * (1.0 - beta_value)
    )

    mean_beta_gradient = (
        mean_raw_gradient
        / sigmoid_scale
    )

    mean_loss = (
        total_loss
        / total_examples
    )

    return (
        mean_raw_gradient.cpu(),
        mean_beta_gradient.cpu(),
        mean_loss,
        total_examples,
        batches_used,
    )


def find_crossings(
    betas,
    gradients,
):
    """
    Find sign changes and classify them.

    negative -> positive:
        local minimum candidate

    positive -> negative:
        local maximum candidate
    """

    crossings = []

    for i in range(
        len(betas) - 1
    ):

        beta_a = betas[i]
        beta_b = betas[i + 1]

        grad_a = gradients[i]
        grad_b = gradients[i + 1]

        if grad_a == 0.0:

            crossings.append({
                "beta":
                    beta_a,
                "type":
                    "exact_zero",
            })

            continue

        if (
            grad_a
            * grad_b
            < 0.0
        ):

            root = (
                beta_a
                - grad_a
                * (
                    beta_b
                    - beta_a
                )
                / (
                    grad_b
                    - grad_a
                )
            )

            if (
                grad_a < 0.0
                and grad_b > 0.0
            ):
                crossing_type = (
                    "minimum_candidate"
                )

            else:
                crossing_type = (
                    "maximum_candidate"
                )

            crossings.append({
                "beta":
                    root,
                "type":
                    crossing_type,
                "left_gradient":
                    grad_a,
                "right_gradient":
                    grad_b,
            })

    return crossings


def main():

    args = get_args()

    fix_seed(args.seed)

    print("=" * 82)
    print(
        "PREDICTIVE SW-CT: "
        "FIXED-CLASSIFIER BETA LANDSCAPE"
    )
    print("=" * 82)

    print(
        "Dataset:",
        args.transfer_ds,
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Probe betas:",
        PROBE_BETAS,
    )

    print(
        "\nClassifier will be trained ONCE "
        "using original ReLU features."
    )

    print(
        "That exact classifier will be frozen "
        "for every beta probe."
    )

    print(
        "Test set is not used."
    )

    dataset = (
        f"{args.pretrained_ds}_to_"
        f"{args.transfer_ds}"
    )

    (
        train_loader,
        test_loader,
        val_loader,
    ) = get_data_loaders(
        dataset,
        seed=args.seed,
        train_batch_size=args.train_bs,
        test_batch_size=args.test_bs,
    )

    # Explicitly do not use test data.
    del test_loader

    # --------------------------------------------------
    # Original pretrained ReLU model
    # --------------------------------------------------

    base_model = (
        get_pretrained_model(
            args.pretrained_ds,
            args.model,
        )
        .to(device)
    )

    # --------------------------------------------------
    # Train ONE classifier
    # --------------------------------------------------

    (
        base_model,
        baseline_val_acc,
        baseline_best_epoch,
    ) = train_baseline_classifier(
        base_model,
        train_loader,
        val_loader,
        args.train_bs,
        args.test_bs,
        args.classifier_epochs,
        args.seed,
    )

    # Everything frozen from here on.
    for param in base_model.parameters():
        param.requires_grad = False

    print("\n" + "=" * 82)
    print("STARTING FIXED-CLASSIFIER SWEEP")
    print("=" * 82)

    records = []

    # --------------------------------------------------
    # Beta sweep
    # --------------------------------------------------

    for beta in PROBE_BETAS:

        print("\n" + "-" * 82)

        print(
            f"BETA = {beta:.2f}"
        )

        print("-" * 82)

        stage_model = (
            replace_resnet_relu_stagewise(
                copy.deepcopy(
                    base_model
                ),
                init_beta=beta,
                coeff=0.5,
            )
            .to(device)
        )

        loaded_betas = (
            get_stage_betas(
                stage_model
            )
        )

        print(
            "Loaded betas:",
            [
                round(x, 6)
                for x in loaded_betas
            ],
        )

        # Same frozen classifier.
        val_loss, val_acc = (
            evaluate_full_model(
                stage_model,
                val_loader,
            )
        )

        (
            raw_gradient,
            beta_gradient,
            mean_train_loss,
            total_examples,
            batches_used,
        ) = measure_beta_gradients(
            stage_model,
            train_loader,
            beta,
            args.seed,
            args.num_gradient_batches,
        )

        print(
            f"Fixed-classifier "
            f"val_acc={val_acc:.2f}%"
        )

        print(
            "dL/dbeta:"
        )

        for stage, value in enumerate(
            beta_gradient.tolist(),
            start=1,
        ):

            print(
                f"  Stage {stage}: "
                f"{value:+.8f}"
            )

        records.append({
            "beta":
                beta,
            "validation_loss":
                val_loss,
            "validation_accuracy":
                val_acc,
            "mean_training_loss":
                mean_train_loss,
            "raw_beta_gradient":
                raw_gradient.tolist(),
            "beta_gradient":
                beta_gradient.tolist(),
            "examples_used":
                total_examples,
            "batches_used":
                batches_used,
        })

        del stage_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 82)
    print("FIXED-CLASSIFIER LANDSCAPE SUMMARY")
    print("=" * 82)

    print(
        f"{'beta':>6} "
        f"{'g1':>12} "
        f"{'g2':>12} "
        f"{'g3':>12} "
        f"{'g4':>12} "
        f"{'val_acc':>10}"
    )

    for record in records:

        g = record[
            "beta_gradient"
        ]

        print(
            f"{record['beta']:>6.2f} "
            f"{g[0]:>+12.6f} "
            f"{g[1]:>+12.6f} "
            f"{g[2]:>+12.6f} "
            f"{g[3]:>+12.6f} "
            f"{record['validation_accuracy']:>9.2f}%"
        )

    # --------------------------------------------------
    # Zero crossings
    # --------------------------------------------------

    print("\n" + "=" * 82)
    print("ZERO-CROSSING ANALYSIS")
    print("=" * 82)

    beta_values = [
        record["beta"]
        for record in records
    ]

    crossing_output = {}

    for stage_index in range(4):

        gradients = [
            record[
                "beta_gradient"
            ][stage_index]
            for record in records
        ]

        crossings = find_crossings(
            beta_values,
            gradients,
        )

        crossing_output[
            str(stage_index + 1)
        ] = crossings

        print(
            f"\nStage {stage_index + 1}:"
        )

        if not crossings:

            closest_index = min(
                range(
                    len(gradients)
                ),
                key=lambda i:
                    abs(
                        gradients[i]
                    ),
            )

            print(
                "  No sign crossing."
            )

            print(
                f"  Smallest |g| at "
                f"beta="
                f"{beta_values[closest_index]:.2f}, "
                f"g="
                f"{gradients[closest_index]:+.8f}"
            )

        else:

            for crossing in crossings:

                print(
                    f"  beta="
                    f"{crossing['beta']:.6f} "
                    f"-> "
                    f"{crossing['type']}"
                )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    os.makedirs(
        "./results",
        exist_ok=True,
    )

    result_path = (
        "./results/"
        f"beta_fixed_classifier_landscape_"
        f"{args.pretrained_ds}_to_"
        f"{args.transfer_ds}_"
        f"{args.model}_"
        f"seed{args.seed}.json"
    )

    output = {
        "dataset":
            args.transfer_ds,
        "seed":
            args.seed,
        "baseline_relu_validation_accuracy":
            baseline_val_acc,
        "baseline_classifier_best_epoch":
            baseline_best_epoch,
        "probe_betas":
            PROBE_BETAS,
        "records":
            records,
        "zero_crossings":
            crossing_output,
        "classifier_retrained_per_beta":
            False,
        "uses_test_set":
            False,
        "uses_078_anchor":
            False,
        "uses_learned_swct_targets":
            False,
    }

    with open(
        result_path,
        "w",
    ) as handle:

        json.dump(
            output,
            handle,
            indent=4,
        )

    print(
        f"\nSaved to: "
        f"{result_path}"
    )

    print("=" * 82)


if __name__ == "__main__":
    main()