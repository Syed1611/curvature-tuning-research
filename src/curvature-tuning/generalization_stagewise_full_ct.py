"""
8-Parameter Stage-Wise Curvature Tuning.

ResNet-18 is frozen except for:
1. The final classifier.
2. Four trainable beta parameters.
3. Four trainable coeff c parameters.

Total curvature parameters = 8.
"""

import os
import copy
import argparse

import torch
import wandb

from torch import nn, optim
from loguru import logger

from utils.data import (
    get_data_loaders,
    DATASET_TO_NUM_CLASSES,
)

from utils.utils import (
    get_pretrained_model,
    get_file_name,
    fix_seed,
    set_logger,
    save_result_json,
)

from utils.curvature_tuning import (
    replace_resnet_relu_stagewise_full,
    get_stage_betas,
    get_stage_coeffs,
)

from train import (
    train_epoch,
    test_epoch,
    WarmUpLR,
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def transfer(
    model,
    train_loader,
    val_loader,
    lr=1e-3,
    ct_lr=1e-2,
):
    criterion = nn.CrossEntropyLoss()

    beta_params = list(
        model.stage_raw_betas.parameters()
    )

    coeff_params = list(
        model.stage_raw_coeffs.parameters()
    )

    curvature_params = (
        beta_params + coeff_params
    )

    curvature_param_ids = {
        id(p) for p in curvature_params
    }

    # Primarily the final classifier.
    other_params = [
        p
        for p in model.parameters()
        if p.requires_grad
        and id(p) not in curvature_param_ids
    ]

    optimizer = torch.optim.Adam([
        {
            "params": curvature_params,
            "lr": ct_lr,
        },
        {
            "params": other_params,
            "lr": lr,
        },
    ])

    warmup_scheduler = WarmUpLR(
        optimizer,
        len(train_loader),
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10],
        gamma=0.1,
    )

    best_model = copy.deepcopy(model)
    best_acc = float("-inf")

    for epoch in range(1, 21):

        train_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            warmup_scheduler,
        )

        _, val_acc = test_epoch(
            epoch,
            model,
            val_loader,
            criterion,
            device,
        )

        current_betas = get_stage_betas(model)
        current_coeffs = get_stage_coeffs(model)

        logger.info(
            f"Epoch {epoch}: "
            f"val_acc={val_acc:.2f}, "
            f"betas="
            f"{[round(x, 4) for x in current_betas]}, "
            f"coeffs="
            f"{[round(x, 4) for x in current_coeffs]}"
        )

        if val_acc > best_acc:
            best_model = copy.deepcopy(model)
            best_acc = val_acc

            logger.info(
                f"New best validation accuracy: "
                f"{val_acc:.2f}% at epoch {epoch}"
            )

        scheduler.step()

    return best_model, best_acc


def get_args():

    parser = argparse.ArgumentParser(
        description=(
            "8-Parameter Stage-Wise "
            "Curvature Tuning"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
        "--transfer_train_bs",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--transfer_test_bs",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--init_beta",
        type=float,
        default=0.78,
    )

    parser.add_argument(
        "--init_coeff",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--ct_lr",
        type=float,
        default=0.01,
    )

    return parser.parse_args()


def main():

    args = get_args()

    if args.model != "resnet18":
        raise ValueError(
            "Start this experiment with "
            "ResNet-18 only."
        )

    transfer_ds_alias = (
        args.transfer_ds.replace("/", "-")
    )

    experiment_id = (
        f"stage_full_ct_"
        f"{args.pretrained_ds}_to_"
        f"{transfer_ds_alias}_"
        f"{args.model}_"
        f"seed{args.seed}_"
        f"initbeta{args.init_beta}_"
        f"initc{args.init_coeff}_"
        f"ctlr{args.ct_lr}"
    )

    result_path = (
        f"./results/{experiment_id}.json"
    )

    if os.path.exists(result_path):
        print(
            f"Result already exists: "
            f"{result_path}"
        )
        return

    f_name = get_file_name(__file__)

    log_file_path = set_logger(
        name=experiment_id
    )

    logger.info(
        f"Log file: {log_file_path}"
    )

    fix_seed(args.seed)

    wandb.init(
        project="stage-wise-full-ct",
        name=experiment_id,
        config=vars(args),
        mode="disabled",
    )

    logger.info(
        f"Running on {device}"
    )

    dataset = (
        f"{args.pretrained_ds}_to_"
        f"{args.transfer_ds}"
    )

    # Load pretrained model.
    model = get_pretrained_model(
        args.pretrained_ds,
        args.model,
    )

    # Freeze pretrained backbone.
    for param in model.parameters():
        param.requires_grad = False

    # New target-dataset classifier.
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=DATASET_TO_NUM_CLASSES[
            args.transfer_ds
        ],
    ).to(device)

    train_loader, test_loader, val_loader = (
        get_data_loaders(
            dataset,
            seed=args.seed,
            train_batch_size=
                args.transfer_train_bs,
            test_batch_size=
                args.transfer_test_bs,
        )
    )

    # Replace ReLUs with 8-parameter
    # Stage-Wise CTUs.
    stage_model = (
        replace_resnet_relu_stagewise_full(
            copy.deepcopy(model),
            init_beta=args.init_beta,
            init_coeff=args.init_coeff,
        ).to(device)
    )

    initial_betas = get_stage_betas(
        stage_model
    )

    initial_coeffs = get_stage_coeffs(
        stage_model
    )

    logger.info(
        f"Initial stage betas: "
        f"{initial_betas}"
    )

    logger.info(
        f"Initial stage coeffs: "
        f"{initial_coeffs}"
    )

    beta_param_count = sum(
        p.numel()
        for p
        in stage_model.stage_raw_betas.parameters()
    )

    coeff_param_count = sum(
        p.numel()
        for p
        in stage_model.stage_raw_coeffs.parameters()
    )

    curvature_params = (
        beta_param_count
        + coeff_param_count
    )

    total_trainable_params = sum(
        p.numel()
        for p in stage_model.parameters()
        if p.requires_grad
    )

    logger.info(
        f"Trainable beta parameters: "
        f"{beta_param_count}"
    )

    logger.info(
        f"Trainable coeff parameters: "
        f"{coeff_param_count}"
    )

    logger.info(
        f"Total trainable curvature parameters: "
        f"{curvature_params}"
    )

    logger.info(
        f"Total trainable parameters: "
        f"{total_trainable_params}"
    )

    assert beta_param_count == 4, (
        f"Expected 4 beta parameters, "
        f"got {beta_param_count}"
    )

    assert coeff_param_count == 4, (
        f"Expected 4 coeff parameters, "
        f"got {coeff_param_count}"
    )

    assert curvature_params == 8, (
        f"Expected 8 curvature parameters, "
        f"got {curvature_params}"
    )

    # Train.
    stage_model, best_val_acc = transfer(
        stage_model,
        train_loader,
        val_loader,
        ct_lr=args.ct_lr,
    )

    criterion = nn.CrossEntropyLoss()

    _, test_acc = test_epoch(
        -1,
        stage_model,
        test_loader,
        criterion,
        device,
    )

    final_betas = get_stage_betas(
        stage_model
    )

    final_coeffs = get_stage_coeffs(
        stage_model
    )

    logger.info(
        f"8-Parameter Stage-Wise CT "
        f"Test Accuracy: {test_acc:.2f}%"
    )

    logger.info(
        f"Final stage betas: "
        f"{final_betas}"
    )

    logger.info(
        f"Final stage coeffs: "
        f"{final_coeffs}"
    )

    os.makedirs(
        "./ckpts",
        exist_ok=True,
    )

    checkpoint_path = (
        f"./ckpts/{experiment_id}.pth"
    )

    torch.save(
        stage_model.state_dict(),
        checkpoint_path,
    )

    save_result_json(
        result_path,
        total_trainable_params,
        test_acc,
        curvature_params=curvature_params,
        beta_params=beta_param_count,
        coeff_params=coeff_param_count,
        initial_beta=args.init_beta,
        initial_coeff=args.init_coeff,
        stage_betas=final_betas,
        stage_coeffs=final_coeffs,
        best_val_acc=best_val_acc,
        ct_lr=args.ct_lr,
    )

    logger.info(
        f"Results saved to {result_path}"
    )

    wandb.finish()


if __name__ == "__main__":
    main()