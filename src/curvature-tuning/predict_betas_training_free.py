"""
Training-Free Stage-Wise Beta Predictor.

Goal:
    Predict four stage-wise beta values BEFORE downstream training.

The predictor performs NO:
    - classifier optimization
    - beta optimization
    - SW-CT training
    - use of learned SW-CT beta targets
    - use of beta=0.78
    - test-set evaluation

Method:
    1. Keep pretrained ResNet-18 frozen.
    2. Curve one stage at a time.
    3. Extract frozen train/validation features.
    4. Build class centroids from train features.
    5. Score each beta using nearest-centroid validation accuracy.
    6. Use quadratic interpolation around the best coarse beta.
    7. Output four predicted betas.

After this script predicts the betas, downstream training
should be run ONCE with those fixed values.
"""

import argparse
import copy
import json
import os

import torch
import torch.nn.functional as F
from torch import nn

from utils.data import get_data_loaders

from utils.utils import (
    get_pretrained_model,
    fix_seed,
)

from beta_stage_isolated_landscape import (
    replace_one_stage,
)


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

    return parser.parse_args()


@torch.inference_mode()
def extract_features(
    model,
    loader,
    seed,
):

    fix_seed(seed)

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

        # Normalize individual feature vectors.
        outputs = F.normalize(
            outputs,
            p=2,
            dim=1,
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


def nearest_centroid_score(
    train_features,
    train_labels,
    val_features,
    val_labels,
):
    """
    Non-parametric classification.

    No classifier is fitted.

    Class representation =
    mean normalized feature vector.
    """

    classes = torch.unique(
        train_labels
    )

    centroids = []

    for class_id in classes:

        class_features = train_features[
            train_labels == class_id
        ]

        centroid = class_features.mean(
            dim=0
        )

        centroid = F.normalize(
            centroid.unsqueeze(0),
            p=2,
            dim=1,
        ).squeeze(0)

        centroids.append(
            centroid
        )

    centroids = torch.stack(
        centroids
    )

    # Features and centroids are normalized,
    # so dot product = cosine similarity.
    similarity = (
        val_features
        @ centroids.T
    )

    predicted_indices = (
        similarity.argmax(
            dim=1
        )
    )

    predictions = classes[
        predicted_indices
    ]

    accuracy = (
        (
            predictions
            == val_labels
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    return accuracy


def quadratic_peak(
    left_beta,
    center_beta,
    right_beta,
    left_score,
    center_score,
    right_score,
):
    """
    Quadratic interpolation around the best
    coarse validation score.

    Prediction is restricted to the local interval.
    """

    h = (
        center_beta
        - left_beta
    )

    denominator = (
        left_score
        - 2.0 * center_score
        + right_score
    )

    # We want a concave local peak.
    # If not concave or essentially flat,
    # keep the best measured beta.
    if denominator >= -1e-12:
        return center_beta

    predicted = (
        center_beta
        + (
            h / 2.0
        )
        * (
            left_score
            - right_score
        )
        / denominator
    )

    # Never extrapolate outside the
    # three points used to fit the curve.
    predicted = max(
        left_beta,
        min(
            right_beta,
            predicted,
        ),
    )

    return predicted


def predict_stage_beta(
    records,
):

    scores = [
        record["score"]
        for record in records
    ]

    best_index = max(
        range(
            len(records)
        ),
        key=lambda i:
            scores[i],
    )

    best_beta = (
        records[
            best_index
        ]["beta"]
    )

    # If the best point lies at a search boundary,
    # use that boundary.
    if best_index == 0:
        return best_beta

    if (
        best_index
        == len(records) - 1
    ):
        return best_beta

    left = records[
        best_index - 1
    ]

    center = records[
        best_index
    ]

    right = records[
        best_index + 1
    ]

    return quadratic_peak(
        left["beta"],
        center["beta"],
        right["beta"],
        left["score"],
        center["score"],
        right["score"],
    )


def main():

    args = get_args()

    fix_seed(
        args.seed
    )

    print("=" * 82)
    print(
        "TRAINING-FREE STAGE-WISE "
        "BETA PREDICTOR"
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
        "Candidate interval:",
        "[0.70, 0.99]",
    )

    print(
        "Probe values:",
        PROBE_BETAS,
    )

    print(
        "\nNo classifier training."
    )

    print(
        "No beta training."
    )

    print(
        "No 0.78 initialization."
    )

    print(
        "No learned SW-CT targets."
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
        train_batch_size=
            args.train_bs,
        test_batch_size=
            args.test_bs,
    )

    # Predictor is forbidden from seeing test set.
    del test_loader

    base_model = (
        get_pretrained_model(
            args.pretrained_ds,
            args.model,
        )
        .to(device)
    )

    # Remove pretrained ImageNet classifier.
    base_model.fc = nn.Identity()

    for param in (
        base_model.parameters()
    ):
        param.requires_grad = False

    base_model.eval()

    all_results = {}
    predicted_betas = []

    # --------------------------------------------------
    # Predict one beta per stage
    # --------------------------------------------------

    for stage_index in range(4):

        print("\n" + "=" * 82)

        print(
            f"STAGE {stage_index + 1}"
        )

        print("=" * 82)

        records = []

        for beta in PROBE_BETAS:

            model = copy.deepcopy(
                base_model
            )

            (
                model,
                relu_count,
            ) = replace_one_stage(
                model,
                stage_index,
                beta,
                coeff=0.5,
            )

            model = model.to(
                device
            )

            train_features, train_labels = (
                extract_features(
                    model,
                    train_loader,
                    args.seed,
                )
            )

            val_features, val_labels = (
                extract_features(
                    model,
                    val_loader,
                    args.seed,
                )
            )

            score = (
                nearest_centroid_score(
                    train_features,
                    train_labels,
                    val_features,
                    val_labels,
                )
            )

            print(
                f"beta={beta:.2f} | "
                f"nearest-centroid "
                f"val_acc={score:.2f}%"
            )

            records.append({
                "beta":
                    beta,
                "score":
                    score,
                "relu_count":
                    relu_count,
            })

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        predicted_beta = (
            predict_stage_beta(
                records
            )
        )

        predicted_betas.append(
            predicted_beta
        )

        all_results[
            str(
                stage_index + 1
            )
        ] = {
            "records":
                records,
            "predicted_beta":
                predicted_beta,
        }

        print(
            f"\nPredicted Stage "
            f"{stage_index + 1} beta: "
            f"{predicted_beta:.6f}"
        )

    # --------------------------------------------------
    # Final prediction
    # --------------------------------------------------

    print("\n" + "=" * 82)
    print(
        "FINAL TRAINING-FREE "
        "BETA PREDICTION"
    )
    print("=" * 82)

    for stage_index, beta in enumerate(
        predicted_betas,
        start=1,
    ):

        print(
            f"Stage {stage_index}: "
            f"{beta:.6f}"
        )

    print(
        "\nPredicted beta vector:"
    )

    print([
        round(
            beta,
            6,
        )
        for beta
        in predicted_betas
    ])

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    os.makedirs(
        "./results",
        exist_ok=True,
    )

    result_path = (
        "./results/"
        f"training_free_beta_prediction_"
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
            PROBE_BETAS,
        "stages":
            all_results,
        "predicted_beta_vector":
            predicted_betas,
        "score":
            "nearest_centroid_validation_accuracy",
        "uses_classifier_training":
            False,
        "uses_beta_training":
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