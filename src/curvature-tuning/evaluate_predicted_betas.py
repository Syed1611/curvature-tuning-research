"""
Evaluate formula-predicted stage-wise beta values.

The four beta values are fixed.
Only a linear classifier is trained.

This tests whether predicted beta values can reproduce
the accuracy benefit of learned SW-CT without beta training.
"""

import argparse
import copy

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
        "--beta1",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--beta2",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--beta3",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--beta4",
        type=float,
        required=True,
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

    return parser.parse_args()


@torch.inference_mode()
def extract_features(
    feature_model,
    loader,
):

    feature_model.eval()

    features = []
    labels = []

    for inputs, targets in loader:

        inputs = inputs.to(device)

        outputs = feature_model(
            inputs
        )

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


def train_classifier(
    model,
    train_loader,
    val_loader,
    test_loader,
    train_bs,
    eval_bs,
    epochs,
):

    # Completely freeze model including betas.
    for param in model.parameters():
        param.requires_grad = False

    # Remove classifier and extract frozen features.
    model.fc = nn.Identity()

    model.eval()

    print("\nExtracting train features...")

    train_features, train_labels = (
        extract_features(
            model,
            train_loader,
        )
    )

    print("Extracting validation features...")

    val_features, val_labels = (
        extract_features(
            model,
            val_loader,
        )
    )

    print("Extracting test features...")

    test_features, test_labels = (
        extract_features(
            model,
            test_loader,
        )
    )

    print(
        "Train features:",
        tuple(train_features.shape),
    )

    # Feature loaders.
    train_dataset = TensorDataset(
        train_features,
        train_labels,
    )

    val_dataset = TensorDataset(
        val_features,
        val_labels,
    )

    test_dataset = TensorDataset(
        test_features,
        test_labels,
    )

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

    num_features = (
        train_features.shape[1]
    )

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
    best_val_acc = 0.0
    best_epoch = 0

    print("\nTraining classifier...")

    for epoch in range(
        1,
        epochs + 1,
    ):

        classifier.train()

        running_loss = 0.0
        correct = 0
        total = 0

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
            100.0
            * correct
            / total
        )

        val_loss, val_acc = (
            evaluate_classifier(
                classifier,
                val_feature_loader,
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

            best_state = copy.deepcopy(
                classifier.state_dict()
            )

        scheduler.step()

    classifier.load_state_dict(
        best_state
    )

    test_loss, test_acc = (
        evaluate_classifier(
            classifier,
            test_feature_loader,
            criterion,
        )
    )

    return (
        best_val_acc,
        best_epoch,
        test_loss,
        test_acc,
    )


def main():

    args = get_args()

    fix_seed(args.seed)

    predicted_betas = [
        args.beta1,
        args.beta2,
        args.beta3,
        args.beta4,
    ]

    print("=" * 72)
    print(
        "FORMULA-PREDICTED SW-CT EVALUATION"
    )
    print("=" * 72)

    print("Device:", device)
    print("Dataset:", args.transfer_ds)
    print("Seed:", args.seed)

    print(
        "Predicted betas:",
        [
            round(x, 6)
            for x in predicted_betas
        ],
    )

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

    model.fc = nn.Linear(
        model.fc.in_features,
        DATASET_TO_NUM_CLASSES[
            args.transfer_ds
        ],
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
    # Insert SW-CT
    # -----------------------------------------------------

    stage_model = (
        replace_resnet_relu_stagewise(
            copy.deepcopy(model),
            init_beta=0.78,
            coeff=0.5,
        )
        .to(device)
    )

    # -----------------------------------------------------
    # Replace beta=0.78 with predicted beta values
    # -----------------------------------------------------

    with torch.no_grad():

        for raw_param, beta in zip(
            stage_model.stage_raw_betas.parameters(),
            predicted_betas,
        ):

            if not (
                0.0 < beta < 1.0
            ):
                raise ValueError(
                    "All betas must be between 0 and 1."
                )

            beta_tensor = torch.tensor(
                beta,
                dtype=raw_param.dtype,
                device=raw_param.device,
            )

            raw_value = torch.logit(
                beta_tensor
            )

            raw_param.copy_(
                raw_value
            )

    actual_betas = get_stage_betas(
        stage_model
    )

    print(
        "Betas loaded into model:",
        [
            round(x, 6)
            for x in actual_betas
        ],
    )

    # Freeze predicted betas.
    for param in (
        stage_model.stage_raw_betas.parameters()
    ):
        param.requires_grad = False

    # -----------------------------------------------------
    # Classifier evaluation
    # -----------------------------------------------------

    (
        best_val_acc,
        best_epoch,
        test_loss,
        test_acc,
    ) = train_classifier(
        stage_model,
        train_loader,
        val_loader,
        test_loader,
        args.train_bs,
        args.test_bs,
        args.classifier_epochs,
    )

    print("\n" + "=" * 72)
    print("FINAL RESULT")
    print("=" * 72)

    print(
        "Fixed predicted betas:",
        [
            round(x, 6)
            for x in actual_betas
        ],
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_acc:.2f}%"
    )

    print(
        f"Best validation epoch: "
        f"{best_epoch}"
    )

    print(
        f"Test loss: "
        f"{test_loss:.6f}"
    )

    print(
        f"Test accuracy: "
        f"{test_acc:.2f}%"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()