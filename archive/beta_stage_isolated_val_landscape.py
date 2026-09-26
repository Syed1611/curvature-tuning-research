"""
Stage-Isolated Beta Landscape Probe.

For each ResNet stage independently:

1. Keep every OTHER stage as the original exact ReLU.
2. Replace ReLUs only in the selected stage with CTUs.
3. Sweep beta over [0.70, 0.99].
4. Reuse one fixed classifier trained on original ReLU features.
5. Measure validation-loss dL/dbeta.
6. Find negative -> positive zero crossings.

No 0.78 anchor.
No learned SW-CT beta targets.
No test-set usage.
"""

import argparse
import copy
import json
import os

import torch
from torch import nn

from utils.data import get_data_loaders

from utils.utils import (
    get_pretrained_model,
    fix_seed,
)

from utils.curvature_tuning import SWCTU

from beta_fixed_classifier_landscape import (
    train_baseline_classifier,
    evaluate_full_model,
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

    return parser.parse_args()


def belongs_to_stage(
    name,
    stage_index,
):

    if stage_index == 0:
        return (
            name == "relu"
            or name.startswith("layer1.")
        )

    if stage_index == 1:
        return name.startswith("layer2.")

    if stage_index == 2:
        return name.startswith("layer3.")

    if stage_index == 3:
        return name.startswith("layer4.")

    raise ValueError(
        f"Invalid stage index: {stage_index}"
    )


def replace_one_stage(
    model,
    stage_index,
    beta,
    coeff=0.5,
):
    """
    Replace ReLUs only in ONE selected stage.

    Every other stage stays as the original exact ReLU.
    """

    if not (
        0.0 < beta < 1.0
    ):
        raise ValueError(
            "beta must be between 0 and 1."
        )

    device_local = next(
        model.parameters()
    ).device

    raw_beta = torch.logit(
        torch.tensor(
            beta,
            dtype=torch.float32,
            device=device_local,
        )
    )

    # One shared beta parameter for this stage.
    model.probe_raw_beta = nn.Parameter(
        raw_beta.clone()
    )

    relu_names = [
        name
        for name, module
        in model.named_modules()
        if isinstance(
            module,
            nn.ReLU,
        )
    ]

    count = 0

    for name in relu_names:

        if not belongs_to_stage(
            name,
            stage_index,
        ):
            continue

        count += 1

        ct = SWCTU(
            shared_raw_beta=
                model.probe_raw_beta,
            coeff=coeff,
        ).to(device_local)

        names = name.split(".")

        parent = model

        for part in names[:-1]:

            if part.isdigit():
                parent = parent[
                    int(part)
                ]

            else:
                parent = getattr(
                    parent,
                    part,
                )

        last = names[-1]

        if last.isdigit():

            parent[
                int(last)
            ] = ct

        else:

            setattr(
                parent,
                last,
                ct,
            )

    if count == 0:
        raise RuntimeError(
            f"No ReLUs found for "
            f"stage {stage_index + 1}"
        )

    return (
        model,
        count,
    )


def measure_single_beta_gradient(
    model,
    train_loader,
    beta,
    seed,
    num_batches=0,
):

    # Freeze everything.
    for param in model.parameters():
        param.requires_grad = False

    # Enable only this stage's beta.
    model.probe_raw_beta.requires_grad = True

    model.eval()

    criterion = nn.CrossEntropyLoss()

    raw_gradient_sum = 0.0

    total_loss = 0.0
    total_examples = 0
    batches_used = 0

    # Reproducible augmentation sequence.
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

        gradient = torch.autograd.grad(
            loss,
            model.probe_raw_beta,
            create_graph=False,
            retain_graph=False,
        )[0]

        batch_size = targets.size(0)

        raw_gradient_sum += (
            gradient.detach().double()
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

    # beta = sigmoid(raw)
    #
    # dL/draw =
    # dL/dbeta * beta*(1-beta)
    beta_gradient = (
        mean_raw_gradient
        / (
            beta
            * (1.0 - beta)
        )
    )

    mean_loss = (
        total_loss
        / total_examples
    )

    return (
        float(
            mean_raw_gradient.cpu()
        ),
        float(
            beta_gradient.cpu()
        ),
        mean_loss,
        total_examples,
        batches_used,
    )


def find_crossings(
    beta_values,
    gradients,
):

    crossings = []

    for i in range(
        len(beta_values) - 1
    ):

        beta_a = beta_values[i]
        beta_b = beta_values[i + 1]

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
            })

    return crossings


def main():

    args = get_args()

    fix_seed(args.seed)

    print("=" * 82)
    print(
        "PREDICTIVE SW-CT: "
        "STAGE-ISOLATED BETA LANDSCAPE"
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
        "\nOnly ONE stage is curved "
        "at a time."
    )

    print(
        "All other stages remain "
        "exact original ReLU."
    )

    print(
        "One fixed baseline classifier "
        "is reused everywhere."
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

    # Never use test data.
    del test_loader

    # --------------------------------------------------
    # Original ReLU model
    # --------------------------------------------------

    base_model = (
        get_pretrained_model(
            args.pretrained_ds,
            args.model,
        )
        .to(device)
    )

    # --------------------------------------------------
    # Train ONE baseline classifier
    # --------------------------------------------------

    (
        base_model,
        baseline_val_acc,
        baseline_epoch,
    ) = train_baseline_classifier(
        base_model,
        train_loader,
        val_loader,
        args.train_bs,
        args.test_bs,
        args.classifier_epochs,
        args.seed,
    )

    for param in (
        base_model.parameters()
    ):
        param.requires_grad = False

    print(
        "\nBaseline ReLU validation "
        f"accuracy: {baseline_val_acc:.2f}%"
    )

    all_stage_results = {}

    predicted_beta_candidates = []

    # --------------------------------------------------
    # Probe each stage independently
    # --------------------------------------------------

    for stage_index in range(4):

        print("\n" + "=" * 82)

        print(
            f"STAGE {stage_index + 1}"
        )

        print("=" * 82)

        stage_records = []

        for beta in PROBE_BETAS:

            probe_model = copy.deepcopy(
                base_model
            )

            (
                probe_model,
                relu_count,
            ) = replace_one_stage(
                probe_model,
                stage_index,
                beta,
                coeff=0.5,
            )

            probe_model = (
                probe_model.to(device)
            )

            (
                val_loss,
                val_acc,
            ) = evaluate_full_model(
                probe_model,
                val_loader,
            )

            (
                raw_gradient,
                beta_gradient,
                val_gradient_loss,
                examples_used,
                batches_used,
            ) = measure_single_beta_gradient(
                probe_model,
                val_loader,
                beta,
                args.seed,
                args.num_gradient_batches,
            )

            print(
                f"beta={beta:.2f} | "
                f"dL/dbeta="
                f"{beta_gradient:+.8f} | "
                f"val_acc="
                f"{val_acc:.2f}%"
            )

            stage_records.append({
                "beta":
                    beta,
                "beta_gradient":
                    beta_gradient,
                "raw_gradient":
                    raw_gradient,
                "validation_loss":
                    val_loss,
                "validation_accuracy":
                    val_acc,
                "validation_gradient_loss":
                    val_gradient_loss,
                "relu_count":
                    relu_count,
                "examples_used":
                    examples_used,
                "batches_used":
                    batches_used,
            })

            del probe_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        beta_values = [
            record["beta"]
            for record in stage_records
        ]

        gradients = [
            record["beta_gradient"]
            for record in stage_records
        ]

        crossings = find_crossings(
            beta_values,
            gradients,
        )

        minimum_candidates = [
            crossing["beta"]
            for crossing in crossings
            if (
                crossing["type"]
                == "minimum_candidate"
            )
        ]

        print(
            f"\nStage {stage_index + 1} "
            f"minimum candidates:"
        )

        if minimum_candidates:

            print([
                round(x, 6)
                for x in minimum_candidates
            ])

        else:

            print(
                "No negative->positive "
                "crossing found."
            )

        all_stage_results[
            str(stage_index + 1)
        ] = {
            "records":
                stage_records,
            "crossings":
                crossings,
            "minimum_candidates":
                minimum_candidates,
        }

        if (
            len(
                minimum_candidates
            )
            == 1
        ):

            predicted_beta_candidates.append(
                minimum_candidates[0]
            )

        else:

            predicted_beta_candidates.append(
                None
            )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n" + "=" * 82)
    print(
        "STAGE-ISOLATED SUMMARY"
    )
    print("=" * 82)

    for stage_index in range(4):

        stage_data = (
            all_stage_results[
                str(
                    stage_index + 1
                )
            ]
        )

        print(
            f"\nStage "
            f"{stage_index + 1}:"
        )

        print(
            f"{'beta':>6} "
            f"{'gradient':>14} "
            f"{'val_acc':>10}"
        )

        for record in (
            stage_data["records"]
        ):

            print(
                f"{record['beta']:>6.2f} "
                f"{record['beta_gradient']:>+14.6f} "
                f"{record['validation_accuracy']:>9.2f}%"
            )

        print(
            "Minimum candidates:",
            [
                round(x, 6)
                for x in
                stage_data[
                    "minimum_candidates"
                ]
            ],
        )

    print(
        "\nAutomatic isolated-stage "
        "beta vector:"
    )

    print([
        (
            round(x, 6)
            if x is not None
            else None
        )
        for x in
        predicted_beta_candidates
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
        f"beta_stage_isolated_"
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
        "baseline_classifier_epoch":
            baseline_epoch,
        "probe_betas":
            PROBE_BETAS,
        "stages":
            all_stage_results,
        "automatic_beta_vector":
            predicted_beta_candidates,
        "other_stages_are_exact_relu":
            True,
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