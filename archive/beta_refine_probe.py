"""
Local refinement of automatically predicted SW-CT betas.

Takes an automatically generated beta vector and:
1. Measures its real four-stage gradient.
2. Perturbs each beta slightly above/below the current value.
3. Estimates a local 4x4 gradient Jacobian.
4. Computes a Newton-like correction.
5. Limits the correction to a small trust region.
6. Constrains beta to [0.70, 0.99].

No learned SW-CT beta target is used.
No 0.78 anchor is used.
"""

import argparse
import copy
import json
import os

import torch

from utils.data import get_data_loaders

from utils.utils import (
    get_pretrained_model,
    fix_seed,
)

from beta_coupled_probe import probe_point


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


BETA_MIN = 0.70
BETA_MAX = 0.99


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

    parser.add_argument(
        "--perturbation",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--trust_radius",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--num_gradient_batches",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main():

    args = get_args()

    fix_seed(args.seed)

    current = torch.tensor(
        [
            args.beta1,
            args.beta2,
            args.beta3,
            args.beta4,
        ],
        dtype=torch.float64,
    )

    print("=" * 78)
    print(
        "PREDICTIVE SW-CT: "
        "LOCAL BETA REFINEMENT"
    )
    print("=" * 78)

    print(
        "Starting automatic beta estimate:",
        [
            round(x, 6)
            for x in current.tolist()
        ],
    )

    print(
        "Perturbation:",
        args.perturbation,
    )

    print(
        "Trust radius:",
        args.trust_radius,
    )

    print(
        "Allowed beta interval:",
        f"[{BETA_MIN}, {BETA_MAX}]",
    )

    # probe_point expects center to exist in args,
    # although the actual beta vector is passed separately.
    args.center = float(
        current.mean().item()
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

    # Test set must not influence prediction.
    del test_loader

    base_model = get_pretrained_model(
        args.pretrained_ds,
        args.model,
    )

    for param in base_model.parameters():
        param.requires_grad = False

    # --------------------------------------------------
    # Gradient at current prediction
    # --------------------------------------------------

    print("\n" + "=" * 78)
    print("CURRENT POINT")
    print("=" * 78)

    (
        center_gradient,
        center_record,
    ) = probe_point(
        base_model,
        current.tolist(),
        train_loader,
        val_loader,
        args,
    )

    center_gradient = (
        center_gradient.double()
    )

    print(
        "\nReal gradient at current point:"
    )

    for i, value in enumerate(
        center_gradient.tolist(),
        start=1,
    ):
        print(
            f"Stage {i}: "
            f"{value:+.8f}"
        )

    # --------------------------------------------------
    # Estimate local 4x4 Jacobian
    # --------------------------------------------------

    jacobian = torch.zeros(
        4,
        4,
        dtype=torch.float64,
    )

    records = [
        {
            "name": "center",
            **center_record,
        }
    ]

    h = args.perturbation

    for stage_index in range(4):

        print("\n" + "=" * 78)
        print(
            f"LOCAL PERTURBATION: "
            f"STAGE {stage_index + 1}"
        )
        print("=" * 78)

        minus_vector = current.clone()
        plus_vector = current.clone()

        minus_vector[stage_index] = max(
            BETA_MIN,
            current[stage_index].item()
            - h,
        )

        plus_vector[stage_index] = min(
            BETA_MAX,
            current[stage_index].item()
            + h,
        )

        (
            gradient_minus,
            record_minus,
        ) = probe_point(
            base_model,
            minus_vector.tolist(),
            train_loader,
            val_loader,
            args,
        )

        (
            gradient_plus,
            record_plus,
        ) = probe_point(
            base_model,
            plus_vector.tolist(),
            train_loader,
            val_loader,
            args,
        )

        gradient_minus = (
            gradient_minus.double()
        )

        gradient_plus = (
            gradient_plus.double()
        )

        denominator = (
            plus_vector[stage_index]
            - minus_vector[stage_index]
        )

        jacobian[
            :,
            stage_index
        ] = (
            gradient_plus
            - gradient_minus
        ) / denominator

        records.append({
            "name":
                f"stage{stage_index + 1}_minus",
            **record_minus,
        })

        records.append({
            "name":
                f"stage{stage_index + 1}_plus",
            **record_plus,
        })

    # --------------------------------------------------
    # Local Newton correction
    # --------------------------------------------------

    print("\n" + "=" * 78)
    print("LOCAL MODEL")
    print("=" * 78)

    print(
        "\nEstimated local Jacobian:"
    )

    for row in jacobian.tolist():

        print(
            " ".join(
                f"{value:+.6f}"
                for value in row
            )
        )

    singular_values = (
        torch.linalg.svdvals(
            jacobian
        )
    )

    condition_number = (
        singular_values.max()
        / singular_values.min()
    ).item()

    print(
        "\nCondition number:",
        round(
            condition_number,
            4,
        ),
    )

    # Solve J * correction = gradient.
    #
    # Newton move is:
    #
    # beta_new = beta_current - correction

    try:

        correction = torch.linalg.solve(
            jacobian,
            center_gradient,
        )

    except RuntimeError:

        correction = (
            torch.linalg.pinv(
                jacobian
            )
            @ center_gradient
        )

    raw_move = (
        -correction
    )

    print(
        "\nRaw Newton move:"
    )

    print([
        round(x, 6)
        for x in raw_move.tolist()
    ])

    # Prevent a local approximation from making
    # another very large jump.
    trusted_move = torch.clamp(
        raw_move,
        min=-args.trust_radius,
        max=args.trust_radius,
    )

    print(
        "\nTrust-region move:"
    )

    print([
        round(x, 6)
        for x in trusted_move.tolist()
    ])

    refined = (
        current
        + trusted_move
    )

    refined = torch.clamp(
        refined,
        min=BETA_MIN,
        max=BETA_MAX,
    )

    print(
        "\nREFINED AUTOMATIC BETAS:"
    )

    print([
        round(x, 6)
        for x in refined.tolist()
    ])

    # Predicted residual according to this
    # local linear model.
    predicted_residual = (
        center_gradient
        + jacobian
        @ (
            refined
            - current
        )
    )

    print(
        "\nPredicted local-gradient residual:"
    )

    print([
        round(x, 8)
        for x
        in predicted_residual.tolist()
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
        f"beta_refine_probe_"
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
        "starting_beta":
            current.tolist(),
        "starting_gradient":
            center_gradient.tolist(),
        "perturbation":
            args.perturbation,
        "trust_radius":
            args.trust_radius,
        "jacobian":
            jacobian.tolist(),
        "condition_number":
            condition_number,
        "raw_newton_move":
            raw_move.tolist(),
        "trusted_move":
            trusted_move.tolist(),
        "refined_beta":
            refined.tolist(),
        "predicted_residual":
            predicted_residual.tolist(),
        "records":
            records,
        "uses_test_set":
            False,
        "uses_078_anchor":
            False,
        "uses_learned_beta_targets":
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

    print("=" * 78)


if __name__ == "__main__":
    main()