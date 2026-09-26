"""
Beta Landscape Probe for Predictive Stage-Wise Curvature Tuning.

Goal:
    Estimate useful stage-wise beta values directly over the interval
    [0.70, 0.99], without assuming beta=0.78.

For each probe beta:
    1. Set ALL four stage betas to that fixed value.
    2. Freeze the pretrained backbone and betas.
    3. Train only a linear classifier.
    4. Freeze the classifier.
    5. Measure dL/dbeta separately for each of the four stages.

Then inspect where each stage gradient crosses zero:

        dL/dbeta_s = 0

A zero crossing is a candidate stationary beta for that stage.

This script uses training data for the gradient measurement
and validation data only for selecting the linear classifier checkpoint.
It does NOT use the test set.
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


DEFAULT_BETAS = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
]


def get_args():

    parser = argparse.ArgumentParser(
        description="Stage-wise beta landscape probe"
    )

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


def train_linear_classifier(
    stage_model,
    train_loader,
    val_loader,
    train_bs,
    val_bs,
    epochs,
    seed,
):
    """
    Train only a linear classifier on fixed CT features.

    The backbone and beta values remain completely fixed.
    """

    for param in stage_model.parameters():
        param.requires_grad = False

    # Remove classifier so ResNet returns 512-d features.
    stage_model.fc = nn.Identity()

    stage_model.eval()

    # Reset RNG so each beta point gets as comparable
    # a feature-extraction pass as possible.
    fix_seed(seed)

    train_features, train_labels = (
        extract_features(
            stage_model,
            train_loader,
        )
    )

    fix_seed(seed)

    val_features, val_labels = (
        extract_features(
            stage_model,
            val_loader,
        )
    )

    train_dataset = TensorDataset(
        train_features,
        train_labels,
    )

    val_dataset = TensorDataset(
        val_features,
        val_labels,
    )

    # Reset again so classifier initialization and
    # feature shuffling are comparable across beta values.
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

    num_features = train_features.shape[1]

    num_classes = (
        int(train_labels.max().item())
        + 1
    )

    classifier = nn.Linear(
        num_features,
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

        _, val_acc = evaluate_classifier(
            classifier,
            val_feature_loader,
            criterion,
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            best_epoch = epoch

            best_state = copy.deepcopy(
                classifier.state_dict()
            )

        scheduler.step()

    if best_state is None:
        raise RuntimeError(
            "Classifier training did not produce "
            "a valid checkpoint."
        )

    classifier.load_state_dict(
        best_state
    )

    stage_model.fc = classifier

    return (
        stage_model,
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
    """
    Measure average loss gradients with respect to
    each of the four beta parameters.

    Returns:
        raw gradients:  dL/d(raw_beta)
        beta gradients: dL/dbeta

    Because:
        beta = sigmoid(raw_beta)

    then:
        dL/d(raw_beta)
        =
        dL/dbeta * beta * (1-beta)
    """

    for param in model.parameters():
        param.requires_grad = False

    beta_params = list(
        model.stage_raw_betas.parameters()
    )

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

    # Keep the gradient-data pass reproducible.
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
            retain_graph=False,
            create_graph=False,
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


def find_zero_crossings(
    beta_values,
    gradient_values,
):
    """
    Find intervals where the beta-space gradient changes sign.

    Linear interpolation inside an interval:

        beta* =
            beta_a
            - g_a * (beta_b-beta_a)/(g_b-g_a)

    No learned beta target is used.
    """

    roots = []

    for index in range(
        len(beta_values) - 1
    ):

        beta_a = beta_values[index]
        beta_b = beta_values[index + 1]

        grad_a = gradient_values[index]
        grad_b = gradient_values[index + 1]

        if grad_a == 0.0:

            roots.append(beta_a)
            continue

        if grad_a * grad_b < 0.0:

            denominator = (
                grad_b
                - grad_a
            )

            root = (
                beta_a
                - grad_a
                * (beta_b - beta_a)
                / denominator
            )

            roots.append(root)

    if gradient_values[-1] == 0.0:
        roots.append(
            beta_values[-1]
        )

    return roots


def main():

    args = get_args()

    if args.model != "resnet18":
        raise ValueError(
            "Start this landscape experiment "
            "with ResNet-18 only."
        )

    probe_betas = DEFAULT_BETAS

    print("=" * 78)
    print(
        "PREDICTIVE SW-CT: "
        "BETA LANDSCAPE PROBE"
    )
    print("=" * 78)

    print("Device:", device)
    print("Dataset:", args.transfer_ds)
    print("Seed:", args.seed)

    print(
        "Probe betas:",
        probe_betas,
    )

    print(
        "\nNo 0.78 anchor is used."
    )

    print(
        "No learned SW-CT beta values "
        "are used for prediction."
    )

    dataset = (
        f"{args.pretrained_ds}_to_"
        f"{args.transfer_ds}"
    )

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------

    fix_seed(args.seed)

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

    # Test loader intentionally not used.
    del test_loader

    # -----------------------------------------------------
    # Base pretrained model
    # -----------------------------------------------------

    fix_seed(args.seed)

    base_model = get_pretrained_model(
        args.pretrained_ds,
        args.model,
    )

    for param in base_model.parameters():
        param.requires_grad = False

    results = []

    # -----------------------------------------------------
    # Landscape
    # -----------------------------------------------------

    for probe_beta in probe_betas:

        print("\n" + "-" * 78)

        print(
            f"PROBE BETA = "
            f"{probe_beta:.2f}"
        )

        print("-" * 78)

        # Fresh copy for every beta point.
        stage_model = (
            replace_resnet_relu_stagewise(
                copy.deepcopy(
                    base_model
                ),
                init_beta=probe_beta,
                coeff=0.5,
            )
            .to(device)
        )

        loaded_betas = get_stage_betas(
            stage_model
        )

        print(
            "Loaded stage betas:",
            [
                round(x, 6)
                for x in loaded_betas
            ],
        )

        # ---------------------------------------------
        # Fit classifier for this fixed beta
        # ---------------------------------------------

        (
            stage_model,
            best_val_acc,
            best_epoch,
        ) = train_linear_classifier(
            stage_model,
            train_loader,
            val_loader,
            args.train_bs,
            args.test_bs,
            args.classifier_epochs,
            args.seed,
        )

        print(
            f"Classifier best val acc: "
            f"{best_val_acc:.2f}% "
            f"(epoch {best_epoch})"
        )

        # ---------------------------------------------
        # Measure beta gradients
        # ---------------------------------------------

        (
            raw_gradient,
            beta_gradient,
            mean_loss,
            total_examples,
            batches_used,
        ) = measure_beta_gradients(
            stage_model,
            train_loader,
            probe_beta,
            args.seed,
            args.num_gradient_batches,
        )

        print(
            "dL/dbeta by stage:"
        )

        for stage_index, value in enumerate(
            beta_gradient.tolist(),
            start=1,
        ):

            direction = (
                "beta should DECREASE"
                if value > 0
                else "beta should INCREASE"
                if value < 0
                else "stationary"
            )

            print(
                f"  Stage {stage_index}: "
                f"{value:+.8f} "
                f"({direction})"
            )

        results.append({
            "beta":
                probe_beta,
            "best_classifier_val_acc":
                best_val_acc,
            "best_classifier_epoch":
                best_epoch,
            "mean_training_loss":
                mean_loss,
            "examples_used":
                total_examples,
            "batches_used":
                batches_used,
            "raw_beta_gradient":
                raw_gradient.tolist(),
            "beta_gradient":
                beta_gradient.tolist(),
        })

        del stage_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------
    # Summary table
    # -----------------------------------------------------

    print("\n" + "=" * 78)
    print("BETA LANDSCAPE SUMMARY")
    print("=" * 78)

    header = (
        f"{'beta':>6} "
        f"{'g1':>12} "
        f"{'g2':>12} "
        f"{'g3':>12} "
        f"{'g4':>12} "
        f"{'val_acc':>10}"
    )

    print(header)

    for result in results:

        gradients = result[
            "beta_gradient"
        ]

        print(
            f"{result['beta']:>6.2f} "
            f"{gradients[0]:>+12.6f} "
            f"{gradients[1]:>+12.6f} "
            f"{gradients[2]:>+12.6f} "
            f"{gradients[3]:>+12.6f} "
            f"{result['best_classifier_val_acc']:>9.2f}%"
        )

    # -----------------------------------------------------
    # Zero-crossing candidates
    # -----------------------------------------------------

    beta_values = [
        result["beta"]
        for result in results
    ]

    stage_root_candidates = {}

    print("\n" + "=" * 78)
    print("ZERO-CROSSING CANDIDATES")
    print("=" * 78)

    for stage_index in range(4):

        stage_gradients = [
            result["beta_gradient"][
                stage_index
            ]
            for result in results
        ]

        roots = find_zero_crossings(
            beta_values,
            stage_gradients,
        )

        stage_root_candidates[
            str(stage_index + 1)
        ] = roots

        if roots:

            print(
                f"Stage {stage_index + 1}: "
                f"{[round(x, 6) for x in roots]}"
            )

        else:

            # Diagnostic only:
            # show where absolute gradient is smallest.
            closest_index = min(
                range(
                    len(stage_gradients)
                ),
                key=lambda i:
                    abs(
                        stage_gradients[i]
                    ),
            )

            closest_beta = (
                beta_values[
                    closest_index
                ]
            )

            closest_gradient = (
                stage_gradients[
                    closest_index
                ]
            )

            print(
                f"Stage {stage_index + 1}: "
                f"NO sign change in sampled range. "
                f"Smallest |gradient| at "
                f"beta={closest_beta:.2f} "
                f"(g={closest_gradient:+.6f})"
            )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    os.makedirs(
        "./results",
        exist_ok=True,
    )

    result_path = (
        "./results/"
        f"beta_landscape_"
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
        "probe_betas":
            probe_betas,
        "results":
            results,
        "zero_crossing_candidates":
            stage_root_candidates,
        "notes": {
            "uses_test_set":
                False,
            "uses_078_anchor":
                False,
            "uses_learned_swct_beta_targets":
                False,
        },
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

    print("=" * 78)


if __name__ == "__main__":
    main()