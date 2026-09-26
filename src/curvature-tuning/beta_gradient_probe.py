"""
Initial beta-gradient probe for Predictive Stage-Wise Curvature Tuning.

Purpose:
1. Load pretrained ResNet-18.
2. Replace ReLUs with four stage-wise CT parameters.
3. Keep beta fixed at the initial value while training only the classifier.
4. Freeze the classifier.
5. Measure the downstream loss gradient with respect to each raw beta.
6. DO NOT update beta.

If gradient < 0:
    gradient descent would increase beta.

If gradient > 0:
    gradient descent would decrease beta.
"""

import argparse
import copy
import json
import os

import torch
from torch import nn

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

from train import linear_probe


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Initial gradient probe for stage-wise beta prediction"
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
        "--num_batches",
        type=int,
        default=0,
        help="0 means use the entire training loader",
    )

    return parser.parse_args()


def main():
    args = get_args()

    if args.model != "resnet18":
        raise ValueError(
            "This first probe is designed for ResNet-18."
        )

    fix_seed(args.seed)

    print("=" * 70)
    print("PREDICTIVE SW-CT: INITIAL BETA GRADIENT PROBE")
    print("=" * 70)
    print("Device:", device)
    print("Dataset:", args.transfer_ds)
    print("Seed:", args.seed)
    print("Initial beta:", args.init_beta)

    dataset = (
        f"{args.pretrained_ds}_to_{args.transfer_ds}"
    )

    # ---------------------------------------------------------
    # 1. Load pretrained model
    # ---------------------------------------------------------
    model = get_pretrained_model(
        args.pretrained_ds,
        args.model,
    )

    # Freeze original pretrained network.
    for param in model.parameters():
        param.requires_grad = False

    # New downstream classifier.
    model.fc = nn.Linear(
        model.fc.in_features,
        DATASET_TO_NUM_CLASSES[args.transfer_ds],
    ).to(device)

    # ---------------------------------------------------------
    # 2. Data
    # ---------------------------------------------------------
    train_loader, test_loader, val_loader = (
        get_data_loaders(
            dataset,
            seed=args.seed,
            train_batch_size=args.train_bs,
            test_batch_size=args.test_bs,
        )
    )

    # ---------------------------------------------------------
    # 3. Insert four SW-CT betas at beta = 0.78
    # ---------------------------------------------------------
    stage_model = replace_resnet_relu_stagewise(
        copy.deepcopy(model),
        init_beta=args.init_beta,
        coeff=0.5,
    ).to(device)

    print(
        "Initial stage betas:",
        [
            round(x, 6)
            for x in get_stage_betas(stage_model)
        ],
    )

    # ---------------------------------------------------------
    # 4. Freeze betas while training classifier
    # ---------------------------------------------------------
    for p in stage_model.stage_raw_betas.parameters():
        p.requires_grad = False

    print("\nTraining classifier with beta fixed...")

    stage_model, best_val_acc = linear_probe(
        stage_model,
        train_loader,
        val_loader,
        new_train_batch_size=args.train_bs,
        new_val_batch_size=args.test_bs,
    )

    print(
        f"Classifier calibration complete. "
        f"Best validation accuracy: {best_val_acc:.2f}%"
    )

    # ---------------------------------------------------------
    # 5. Freeze everything
    # ---------------------------------------------------------
    for p in stage_model.parameters():
        p.requires_grad = False

    # Re-enable ONLY the four raw-beta parameters.
    for p in stage_model.stage_raw_betas.parameters():
        p.requires_grad = True

    trainable = [
        name
        for name, p in stage_model.named_parameters()
        if p.requires_grad
    ]

    print("\nTrainable parameters during gradient probe:")
    for name in trainable:
        print(" ", name)

    if len(trainable) != 4:
        raise RuntimeError(
            f"Expected exactly 4 trainable beta parameters, "
            f"found {len(trainable)}"
        )

    # ---------------------------------------------------------
    # 6. Measure gradients
    # ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    # Eval mode keeps pretrained BN statistics fixed.
    stage_model.eval()

    grad_sum = torch.zeros(
        4,
        dtype=torch.float64,
        device=device,
    )

    loss_sum = 0.0
    total_examples = 0
    batches_used = 0

    for inputs, targets in train_loader:

        if (
            args.num_batches > 0
            and batches_used >= args.num_batches
        ):
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        stage_model.zero_grad(set_to_none=True)

        outputs = stage_model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()

        batch_size = targets.size(0)

        batch_grads = torch.stack([
            p.grad.detach().double()
            for p
            in stage_model.stage_raw_betas.parameters()
        ])

        # Weight by batch size so the final number is
        # an example-weighted mean gradient.
        grad_sum += batch_grads * batch_size

        loss_sum += loss.item() * batch_size
        total_examples += batch_size
        batches_used += 1

    if total_examples == 0:
        raise RuntimeError("No training examples were processed.")

    mean_raw_grad = (
        grad_sum / total_examples
    ).cpu()

    mean_loss = loss_sum / total_examples

    # Since:
    #
    # beta = sigmoid(raw_beta)
    #
    # dL/draw_beta =
    # dL/dbeta * beta * (1-beta)
    #
    # This converts the gradient into beta-space
    # for easier interpretation.
    beta_scale = (
        args.init_beta
        * (1.0 - args.init_beta)
    )

    mean_beta_grad = (
        mean_raw_grad / beta_scale
    )

    # ---------------------------------------------------------
    # 7. Interpret directions
    # ---------------------------------------------------------
    directions = []

    for grad in mean_raw_grad.tolist():
        if grad < 0:
            directions.append("UP")
        elif grad > 0:
            directions.append("DOWN")
        else:
            directions.append("FLAT")

    print("\n" + "=" * 70)
    print("GRADIENT PROBE RESULTS")
    print("=" * 70)

    print(
        f"Examples used: {total_examples}"
    )
    print(
        f"Batches used: {batches_used}"
    )
    print(
        f"Mean loss: {mean_loss:.6f}"
    )

    print("\nMean gradients with respect to RAW beta:")

    for i, grad in enumerate(
        mean_raw_grad.tolist(),
        start=1,
    ):
        print(
            f"Stage {i}: {grad:+.8f}"
        )

    print("\nApproximate gradients with respect to beta:")

    for i, grad in enumerate(
        mean_beta_grad.tolist(),
        start=1,
    ):
        print(
            f"Stage {i}: {grad:+.8f}"
        )

    print("\nPredicted beta movement under gradient descent:")

    for i, direction in enumerate(
        directions,
        start=1,
    ):
        print(
            f"Stage {i}: {direction}"
        )

    print("\nExpected learned Beans seed-42 movement:")

    learned_beans_seed42 = [
        0.8154419,
        0.8786671,
        0.8259457,
        0.6574407,
    ]

    expected_directions = []

    for beta in learned_beans_seed42:
        if beta > args.init_beta:
            expected_directions.append("UP")
        elif beta < args.init_beta:
            expected_directions.append("DOWN")
        else:
            expected_directions.append("FLAT")

    for i, (beta, direction) in enumerate(
        zip(
            learned_beans_seed42,
            expected_directions,
        ),
        start=1,
    ):
        print(
            f"Stage {i}: {direction} "
            f"(learned beta={beta:.6f})"
        )

    matches = [
        predicted == expected
        for predicted, expected
        in zip(
            directions,
            expected_directions,
        )
    ]

    print("\nDirection matches:")

    for i, match in enumerate(
        matches,
        start=1,
    ):
        print(
            f"Stage {i}: "
            f"{'YES' if match else 'NO'}"
        )

    print(
        f"\nTotal direction matches: "
        f"{sum(matches)}/4"
    )

    # ---------------------------------------------------------
    # 8. Save diagnostic result
    # ---------------------------------------------------------
    os.makedirs(
        "./results",
        exist_ok=True,
    )

    transfer_ds_alias = (
        args.transfer_ds.replace("/", "-")
    )

    result_path = (
        f"./results/beta_gradient_probe_"
        f"{args.pretrained_ds}_to_"
        f"{transfer_ds_alias}_"
        f"{args.model}_seed{args.seed}.json"
    )

    result = {
        "dataset": args.transfer_ds,
        "seed": args.seed,
        "initial_beta": args.init_beta,
        "examples_used": total_examples,
        "batches_used": batches_used,
        "mean_loss": mean_loss,
        "mean_raw_beta_gradient":
            mean_raw_grad.tolist(),
        "mean_beta_gradient":
            mean_beta_grad.tolist(),
        "predicted_directions":
            directions,
        "learned_swct_betas":
            learned_beans_seed42,
        "expected_directions":
            expected_directions,
        "direction_matches":
            matches,
        "num_direction_matches":
            int(sum(matches)),
        "best_classifier_val_acc":
            best_val_acc,
    }

    with open(
        result_path,
        "w",
    ) as f:
        json.dump(
            result,
            f,
            indent=4,
        )

    print(
        f"\nSaved result to: {result_path}"
    )

    print("\n" + "=" * 70)

    if sum(matches) >= 3:
        print(
            "GOOD SIGN: the initial loss gradient "
            "mostly agrees with learned SW-CT movement."
        )
    else:
        print(
            "The simple initial gradient does not explain "
            "the learned beta directions well enough yet."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()