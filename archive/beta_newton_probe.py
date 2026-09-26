"""
Second-order beta predictor for Predictive SW-CT.

Steps:
1. Load pretrained ResNet-18.
2. Insert four SW-CT betas fixed at beta=0.78.
3. Extract frozen features with the backbone in eval mode.
4. Train only a linear classifier.
5. Freeze the classifier.
6. Compute:
       g = gradient of loss w.r.t. four raw beta parameters
       H = 4x4 Hessian of loss w.r.t. four raw beta parameters
7. Predict beta values using a damped Newton step:

       r_hat = r0 - (H + lambda I)^(-1) g
       beta_hat = sigmoid(r_hat)

No beta optimization is performed.
"""

import argparse
import copy
import json
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from utils.data import (
    get_data_loaders,
    DATASET_TO_NUM_CLASSES,
)

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


BEANS_TARGETS = {
    42: [
        0.8154419,
        0.8786671,
        0.8259457,
        0.6574407,
    ],
    43: [
        0.81150,
        0.88751,
        0.80220,
        0.64478,
    ],
    44: [
        0.81520,
        0.87503,
        0.80709,
        0.63821,
    ],
}


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
        "--init_beta",
        type=float,
        default=0.78,
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
        "--num_probe_batches",
        type=int,
        default=0,
        help="0 = use entire training set for gradient/Hessian",
    )

    return parser.parse_args()


@torch.inference_mode()
def extract_features(
    feature_model,
    loader,
):
    """
    Extract frozen features with the entire backbone
    permanently in eval mode.
    """

    feature_model.eval()

    features = []
    labels = []

    for inputs, targets in loader:

        inputs = inputs.to(device)

        outputs = feature_model(inputs)

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

            outputs = classifier(features)

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


def train_fixed_beta_classifier(
    stage_model,
    train_loader,
    val_loader,
    train_bs,
    val_bs,
    epochs,
):
    """
    This deliberately mimics linear probing:

    - backbone remains in eval mode
    - beta remains fixed
    - features are extracted once
    - only a new linear classifier is trained
    """

    # Freeze every parameter.
    for param in stage_model.parameters():
        param.requires_grad = False

    # Replace FC temporarily with Identity so the model
    # returns the frozen 512-dimensional ResNet features.
    stage_model.fc = nn.Identity()

    stage_model.eval()

    print("\nExtracting frozen train features...")

    train_features, train_labels = (
        extract_features(
            stage_model,
            train_loader,
        )
    )

    print("Extracting frozen validation features...")

    val_features, val_labels = (
        extract_features(
            stage_model,
            val_loader,
        )
    )

    print(
        "Train feature shape:",
        tuple(train_features.shape),
    )

    print(
        "Validation feature shape:",
        tuple(val_features.shape),
    )

    train_dataset = TensorDataset(
        train_features,
        train_labels,
    )

    val_dataset = TensorDataset(
        val_features,
        val_labels,
    )

    classifier_train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=2,
    )

    classifier_val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=2,
    )

    num_features = train_features.shape[1]

    num_classes = (
        train_labels.max().item()
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
        len(classifier_train_loader),
    )

    scheduler = (
        optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[10, 20],
            gamma=0.1,
        )
    )

    best_classifier_state = None
    best_val_acc = 0.0
    best_epoch = None

    print(
        "\nTraining classifier with "
        "beta fixed at 0.78..."
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        classifier.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for features, targets in (
            classifier_train_loader
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

            batch_size = targets.size(0)

            running_loss += (
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

        train_loss = (
            running_loss / total
        )

        train_acc = (
            100.0 * correct / total
        )

        val_loss, val_acc = (
            evaluate_classifier(
                classifier,
                classifier_val_loader,
                criterion,
            )
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.6f} | "
            f"val_acc={val_acc:.2f}%"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            best_epoch = epoch

            best_classifier_state = (
                copy.deepcopy(
                    classifier.state_dict()
                )
            )

        scheduler.step()

    classifier.load_state_dict(
        best_classifier_state
    )

    # Put the trained classifier back into ResNet.
    stage_model.fc = classifier

    print(
        f"\nBest classifier validation "
        f"accuracy: {best_val_acc:.2f}% "
        f"at epoch {best_epoch}"
    )

    return (
        stage_model,
        best_val_acc,
        best_epoch,
    )


def compute_gradient_and_hessian(
    model,
    loader,
    num_batches=0,
):
    """
    Compute the mean gradient and full 4x4 Hessian of
    training loss w.r.t. the four RAW beta parameters.
    """

    # Freeze everything.
    for param in model.parameters():
        param.requires_grad = False

    beta_params = list(
        model.stage_raw_betas.parameters()
    )

    # Re-enable only the four raw beta parameters.
    for param in beta_params:
        param.requires_grad = True

    model.eval()

    criterion = nn.CrossEntropyLoss()

    grad_sum = torch.zeros(
        4,
        dtype=torch.float64,
        device=device,
    )

    hessian_sum = torch.zeros(
        4,
        4,
        dtype=torch.float64,
        device=device,
    )

    loss_sum = 0.0
    total_examples = 0
    batches_used = 0

    print(
        "\nComputing gradient + "
        "4x4 Hessian..."
    )

    for batch_idx, (
        inputs,
        targets,
    ) in enumerate(loader):

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

        # First derivative.
        first_grads = torch.autograd.grad(
            loss,
            beta_params,
            create_graph=True,
            retain_graph=True,
        )

        grad_vector = torch.stack(
            first_grads
        )

        # Full 4x4 second derivative matrix.
        hessian_rows = []

        for i in range(4):

            second_grads = (
                torch.autograd.grad(
                    first_grads[i],
                    beta_params,
                    retain_graph=True,
                    allow_unused=True,
                )
            )

            row = []

            for param, value in zip(
                beta_params,
                second_grads,
            ):

                if value is None:
                    row.append(
                        torch.zeros_like(
                            param
                        )
                    )
                else:
                    row.append(value)

            hessian_rows.append(
                torch.stack(row)
            )

        batch_hessian = torch.stack(
            hessian_rows
        )

        batch_size = targets.size(0)

        grad_sum += (
            grad_vector.detach().double()
            * batch_size
        )

        hessian_sum += (
            batch_hessian.detach().double()
            * batch_size
        )

        loss_sum += (
            loss.item()
            * batch_size
        )

        total_examples += batch_size
        batches_used += 1

        if (
            batches_used == 1
            or batches_used % 10 == 0
        ):
            print(
                f"Processed "
                f"{batches_used} batches"
            )

    mean_gradient = (
        grad_sum
        / total_examples
    )

    mean_hessian = (
        hessian_sum
        / total_examples
    )

    # Numerical Hessian should be symmetric.
    mean_hessian = (
        0.5
        * (
            mean_hessian
            + mean_hessian.T
        )
    )

    mean_loss = (
        loss_sum
        / total_examples
    )

    return (
        mean_gradient.cpu(),
        mean_hessian.cpu(),
        mean_loss,
        total_examples,
        batches_used,
    )


def newton_predict(
    raw_beta_initial,
    gradient,
    hessian,
    damping,
):
    """
    r_hat = r0 - (H + lambda I)^(-1) g
    beta_hat = sigmoid(r_hat)
    """

    dimension = gradient.numel()

    identity = torch.eye(
        dimension,
        dtype=torch.float64,
    )

    matrix = (
        hessian
        + damping * identity
    )

    try:

        step = torch.linalg.solve(
            matrix,
            gradient,
        )

    except RuntimeError:

        # Fallback if matrix is singular.
        step = (
            torch.linalg.pinv(
                matrix
            )
            @ gradient
        )

    predicted_raw = (
        raw_beta_initial
        - step
    )

    predicted_beta = (
        torch.sigmoid(
            predicted_raw
        )
    )

    return (
        predicted_raw,
        predicted_beta,
        step,
    )


def main():

    args = get_args()

    fix_seed(args.seed)

    print("=" * 72)
    print(
        "PREDICTIVE SW-CT: "
        "GRADIENT + HESSIAN PROBE"
    )
    print("=" * 72)

    print("Device:", device)
    print("Dataset:", args.transfer_ds)
    print("Seed:", args.seed)
    print("Initial beta:", args.init_beta)

    dataset = (
        f"{args.pretrained_ds}_to_"
        f"{args.transfer_ds}"
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = get_pretrained_model(
        args.pretrained_ds,
        args.model,
    )

    for param in model.parameters():
        param.requires_grad = False

    num_classes = (
        DATASET_TO_NUM_CLASSES[
            args.transfer_ds
        ]
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    model = model.to(device)

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # SW-CT insertion
    # -----------------------------------------------------

    stage_model = (
        replace_resnet_relu_stagewise(
            copy.deepcopy(model),
            init_beta=args.init_beta,
            coeff=0.5,
        )
        .to(device)
    )

    print(
        "Initial betas:",
        [
            round(x, 6)
            for x in get_stage_betas(
                stage_model
            )
        ],
    )

    # -----------------------------------------------------
    # Fixed-beta linear probe
    # -----------------------------------------------------

    (
        stage_model,
        best_val_acc,
        best_epoch,
    ) = train_fixed_beta_classifier(
        stage_model,
        train_loader,
        val_loader,
        args.train_bs,
        args.test_bs,
        args.classifier_epochs,
    )

    print(
        "\nBetas after classifier training:",
        [
            round(x, 6)
            for x in get_stage_betas(
                stage_model
            )
        ],
    )

    # Must still be exactly 0.78.
    for beta in get_stage_betas(
        stage_model
    ):
        assert abs(
            beta
            - args.init_beta
        ) < 1e-5

    # -----------------------------------------------------
    # Gradient + Hessian
    # -----------------------------------------------------

    (
        gradient,
        hessian,
        mean_loss,
        total_examples,
        batches_used,
    ) = compute_gradient_and_hessian(
        stage_model,
        train_loader,
        args.num_probe_batches,
    )

    raw_beta_initial = torch.stack([
        p.detach().double().cpu()
        for p
        in stage_model.stage_raw_betas.parameters()
    ])

    print("\n" + "=" * 72)
    print("GRADIENT / HESSIAN RESULTS")
    print("=" * 72)

    print(
        f"\nMean probe loss: "
        f"{mean_loss:.6f}"
    )

    print(
        f"Examples used: "
        f"{total_examples}"
    )

    print(
        f"Batches used: "
        f"{batches_used}"
    )

    print(
        "\nMean RAW-beta gradient:"
    )

    for index, value in enumerate(
        gradient.tolist(),
        start=1,
    ):
        print(
            f"Stage {index}: "
            f"{value:+.8f}"
        )

    print("\n4x4 Hessian:")

    for row in hessian.tolist():
        print(
            " ".join(
                f"{value:+.8f}"
                for value in row
            )
        )

    eigenvalues = torch.linalg.eigvalsh(
        hessian
    )

    print(
        "\nHessian eigenvalues:"
    )

    print([
        round(x, 8)
        for x in eigenvalues.tolist()
    ])

    # -----------------------------------------------------
    # Newton predictions
    # -----------------------------------------------------

    dampings = [
        0.0,
        0.001,
        0.01,
        0.05,
        0.1,
        0.5,
        1.0,
    ]

    target = None

    if (
        args.transfer_ds == "beans"
        and args.seed in BEANS_TARGETS
    ):
        target = torch.tensor(
            BEANS_TARGETS[args.seed],
            dtype=torch.float64,
        )

        print(
            "\nActual learned SW-CT target:"
        )

        print([
            round(x, 6)
            for x in target.tolist()
        ])

    print(
        "\nNewton beta predictions:"
    )

    prediction_records = []

    for damping in dampings:

        (
            predicted_raw,
            predicted_beta,
            step,
        ) = newton_predict(
            raw_beta_initial,
            gradient,
            hessian,
            damping,
        )

        beta_list = (
            predicted_beta.tolist()
        )

        record = {
            "damping": damping,
            "predicted_beta":
                beta_list,
            "newton_step":
                step.tolist(),
        }

        if target is not None:

            mae = torch.mean(
                torch.abs(
                    predicted_beta
                    - target
                )
            ).item()

            record["target_mae"] = mae

            print(
                f"lambda={damping:<5} "
                f"beta="
                f"{[round(x, 4) for x in beta_list]} "
                f"MAE={mae:.5f}"
            )

        else:

            print(
                f"lambda={damping:<5} "
                f"beta="
                f"{[round(x, 4) for x in beta_list]}"
            )

        prediction_records.append(
            record
        )

    if target is not None:

        best_record = min(
            prediction_records,
            key=lambda x:
                x["target_mae"],
        )

        print(
            "\nBest Beans-development "
            "damping:"
        )

        print(
            "lambda =",
            best_record["damping"],
        )

        print(
            "predicted beta =",
            [
                round(x, 6)
                for x in
                best_record[
                    "predicted_beta"
                ]
            ],
        )

        print(
            "beta MAE =",
            round(
                best_record[
                    "target_mae"
                ],
                6,
            ),
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
        f"beta_newton_probe_"
        f"{args.pretrained_ds}_to_"
        f"{args.transfer_ds}_"
        f"{args.model}_"
        f"seed{args.seed}.json"
    )

    result = {
        "dataset":
            args.transfer_ds,
        "seed":
            args.seed,
        "initial_beta":
            args.init_beta,
        "best_classifier_val_acc":
            best_val_acc,
        "best_classifier_epoch":
            best_epoch,
        "mean_probe_loss":
            mean_loss,
        "examples_used":
            total_examples,
        "batches_used":
            batches_used,
        "raw_beta_gradient":
            gradient.tolist(),
        "hessian":
            hessian.tolist(),
        "hessian_eigenvalues":
            eigenvalues.tolist(),
        "predictions":
            prediction_records,
    }

    with open(
        result_path,
        "w",
    ) as handle:

        json.dump(
            result,
            handle,
            indent=4,
        )

    print(
        f"\nSaved to: "
        f"{result_path}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()