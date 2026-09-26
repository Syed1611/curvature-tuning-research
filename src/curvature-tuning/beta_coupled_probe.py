"""
Coupled beta predictor for Predictive SW-CT.

No 0.78 anchor.
No learned beta targets.
No alpha scaling.

We estimate the local 4D gradient field:

    g(beta) ~= g(center) + J (beta - center)

where J is a 4x4 Jacobian describing interactions
between the four stage-wise beta values.

Then solve:

    g(beta*) = 0

giving:

    beta* = center - J^{-1} g(center)

Finally constrain predictions to [0.70, 0.99].
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

from utils.curvature_tuning import (
    replace_resnet_relu_stagewise,
    get_stage_betas,
)

from beta_landscape_probe import (
    train_linear_classifier,
    measure_beta_gradients,
)


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
    )

    parser.add_argument(
        "--center",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=0.10,
    )

    return parser.parse_args()


def set_stage_betas(
    model,
    beta_values,
):
    """
    Directly load four fixed beta values into the model.
    """

    if len(beta_values) != 4:
        raise ValueError(
            "Expected exactly four beta values."
        )

    with torch.no_grad():

        for raw_param, beta in zip(
            model.stage_raw_betas.parameters(),
            beta_values,
        ):

            if not (
                0.0 < beta < 1.0
            ):
                raise ValueError(
                    f"Invalid beta: {beta}"
                )

            beta_tensor = torch.tensor(
                beta,
                dtype=raw_param.dtype,
                device=raw_param.device,
            )

            raw_param.copy_(
                torch.logit(
                    beta_tensor
                )
            )


def probe_point(
    base_model,
    beta_vector,
    train_loader,
    val_loader,
    args,
):
    """
    For one 4D beta vector:

    1. Load fixed betas.
    2. Fit classifier.
    3. Measure four beta gradients.
    """

    stage_model = (
        replace_resnet_relu_stagewise(
            copy.deepcopy(
                base_model
            ),
            init_beta=args.center,
            coeff=0.5,
        )
        .to(device)
    )

    set_stage_betas(
        stage_model,
        beta_vector,
    )

    loaded = get_stage_betas(
        stage_model
    )

    print(
        "\nProbe beta vector:",
        [
            round(x, 4)
            for x in loaded
        ],
    )

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

    (
        raw_gradient,
        beta_gradient,
        mean_loss,
        total_examples,
        batches_used,
    ) = measure_beta_gradients(
        stage_model,
        train_loader,
        beta_value=args.center,
        seed=args.seed,
        num_batches=args.num_gradient_batches,
    )

    # Important:
    # measure_beta_gradients converts raw gradient using
    # one scalar beta_value. Because our four betas differ,
    # recompute dL/dbeta correctly stage by stage.

    raw_gradient = raw_gradient.double()

    beta_tensor = torch.tensor(
        beta_vector,
        dtype=torch.float64,
    )

    scales = (
        beta_tensor
        * (1.0 - beta_tensor)
    )

    beta_gradient = (
        raw_gradient
        / scales
    )

    print(
        "dL/dbeta:",
        [
            round(x, 6)
            for x in beta_gradient.tolist()
        ],
    )

    print(
        f"Best val acc: "
        f"{best_val_acc:.2f}% "
        f"(epoch {best_epoch})"
    )

    record = {
        "beta_vector":
            list(beta_vector),
        "raw_gradient":
            raw_gradient.tolist(),
        "beta_gradient":
            beta_gradient.tolist(),
        "best_val_acc":
            best_val_acc,
        "best_epoch":
            best_epoch,
        "mean_loss":
            mean_loss,
        "examples_used":
            total_examples,
        "batches_used":
            batches_used,
    }

    del stage_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        beta_gradient,
        record,
    )


def main():

    args = get_args()

    fix_seed(
        args.seed
    )

    center = args.center
    delta = args.delta

    low = (
        center
        - delta
    )

    high = (
        center
        + delta
    )

    if low < 0.70:
        raise ValueError(
            "Lower probe point fell below 0.70."
        )

    if high > 0.99:
        raise ValueError(
            "Upper probe point exceeded 0.99."
        )

    print("=" * 78)
    print(
        "PREDICTIVE SW-CT: "
        "COUPLED 4D BETA PROBE"
    )
    print("=" * 78)

    print(
        "Dataset:",
        args.transfer_ds,
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Center:",
        center,
    )

    print(
        "Delta:",
        delta,
    )

    print(
        "Probe range:",
        low,
        "to",
        high,
    )

    print(
        "\nNo 0.78 anchor."
    )

    print(
        "No learned SW-CT beta targets."
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

    # Test set is deliberately unused.
    del test_loader

    base_model = (
        get_pretrained_model(
            args.pretrained_ds,
            args.model,
        )
    )

    for param in (
        base_model.parameters()
    ):
        param.requires_grad = False

    # --------------------------------------------------
    # Center point
    # --------------------------------------------------

    center_vector = [
        center,
        center,
        center,
        center,
    ]

    print(
        "\n" + "=" * 78
    )

    print(
        "CENTER PROBE"
    )

    print(
        "=" * 78
    )

    (
        center_gradient,
        center_record,
    ) = probe_point(
        base_model,
        center_vector,
        train_loader,
        val_loader,
        args,
    )

    records = [
        {
            "name":
                "center",
            **center_record,
        }
    ]

    # --------------------------------------------------
    # Estimate 4x4 Jacobian
    # --------------------------------------------------

    jacobian = torch.zeros(
        4,
        4,
        dtype=torch.float64,
    )

    for stage_index in range(4):

        print(
            "\n" + "=" * 78
        )

        print(
            f"PERTURB STAGE "
            f"{stage_index + 1}"
        )

        print(
            "=" * 78
        )

        minus_vector = (
            center_vector.copy()
        )

        plus_vector = (
            center_vector.copy()
        )

        minus_vector[
            stage_index
        ] = low

        plus_vector[
            stage_index
        ] = high

        (
            gradient_minus,
            record_minus,
        ) = probe_point(
            base_model,
            minus_vector,
            train_loader,
            val_loader,
            args,
        )

        (
            gradient_plus,
            record_plus,
        ) = probe_point(
            base_model,
            plus_vector,
            train_loader,
            val_loader,
            args,
        )

        records.append({
            "name":
                f"stage"
                f"{stage_index + 1}"
                f"_minus",
            **record_minus,
        })

        records.append({
            "name":
                f"stage"
                f"{stage_index + 1}"
                f"_plus",
            **record_plus,
        })

        # Central finite difference:
        #
        # dg_i / dbeta_j

        jacobian[
            :,
            stage_index
        ] = (
            gradient_plus
            - gradient_minus
        ) / (
            2.0 * delta
        )

    # --------------------------------------------------
    # Solve g(beta*) = 0
    # --------------------------------------------------

    print(
        "\n" + "=" * 78
    )

    print(
        "COUPLED MODEL"
    )

    print(
        "=" * 78
    )

    print(
        "\nCenter gradient:"
    )

    print([
        round(x, 8)
        for x
        in center_gradient.tolist()
    ])

    print(
        "\nEstimated 4x4 Jacobian:"
    )

    for row in (
        jacobian.tolist()
    ):

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
        "\nJacobian condition number:",
        round(
            condition_number,
            4,
        ),
    )

    center_tensor = torch.tensor(
        center_vector,
        dtype=torch.float64,
    )

    try:

        correction = (
            torch.linalg.solve(
                jacobian,
                center_gradient,
            )
        )

    except RuntimeError:

        print(
            "\nJacobian singular; "
            "using pseudoinverse."
        )

        correction = (
            torch.linalg.pinv(
                jacobian
            )
            @ center_gradient
        )

    predicted_unclipped = (
        center_tensor
        - correction
    )

    predicted_clipped = (
        torch.clamp(
            predicted_unclipped,
            min=0.70,
            max=0.99,
        )
    )

    print(
        "\nPredicted beta "
        "(unclipped):"
    )

    print([
        round(x, 6)
        for x
        in predicted_unclipped.tolist()
    ])

    print(
        "\nPredicted beta "
        "(constrained to "
        "[0.70, 0.99]):"
    )

    print([
        round(x, 6)
        for x
        in predicted_clipped.tolist()
    ])

    # Predicted residual under local model.
    residual = (
        center_gradient
        + jacobian
        @ (
            predicted_clipped
            - center_tensor
        )
    )

    print(
        "\nPredicted gradient residual:"
    )

    print([
        round(x, 8)
        for x
        in residual.tolist()
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
        f"beta_coupled_probe_"
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
        "center":
            center,
        "delta":
            delta,
        "probe_low":
            low,
        "probe_high":
            high,
        "records":
            records,
        "center_gradient":
            center_gradient.tolist(),
        "jacobian":
            jacobian.tolist(),
        "condition_number":
            condition_number,
        "predicted_unclipped":
            predicted_unclipped.tolist(),
        "predicted_clipped":
            predicted_clipped.tolist(),
        "predicted_gradient_residual":
            residual.tolist(),
        "uses_078_anchor":
            False,
        "uses_learned_beta_targets":
            False,
        "uses_test_set":
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

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()