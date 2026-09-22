"""
Stage-Wise Curvature Tuning experiment.

ResNet-18 is frozen except for:
1. The final classifier.
2. Four trainable beta parameters, one per ResNet stage.

c remains fixed at 0.5.
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
    replace_resnet_relu_stagewise,
    get_stage_betas,
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
    beta_lr=1e-1,
):
    criterion = nn.CrossEntropyLoss()

    # The four beta parameters.
    beta_params = list(model.stage_raw_betas.parameters())

    beta_param_ids = {
        id(p) for p in beta_params
    }

    # Normally this is mainly the final classifier.
    other_params = [
        p
        for p in model.parameters()
        if p.requires_grad
        and id(p) not in beta_param_ids
    ]

    optimizer = torch.optim.Adam([
        {
            "params": beta_params,
            "lr": beta_lr,
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

    criterion = nn.CrossEntropyLoss()

    best_model = None
    best_acc = 0.0

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

        logger.info(
            f"Epoch {epoch}: "
            f"val_acc={val_acc:.2f}, "
            f"betas={[round(b, 4) for b in current_betas]}"
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
        description="Stage-Wise Curvature Tuning"
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
        default=0.8,
    )

    parser.add_argument(
        "--beta_lr",
        type=float,
        default=1e-1,
    )

    return parser.parse_args()


def main():

    args = get_args()

    if args.model != "resnet18":
        raise ValueError(
            "Start this experiment with ResNet-18 only."
        )

    transfer_ds_alias = (
        args.transfer_ds.replace("/", "-")
    )

    result_path = (
        f"./results/stage_ct_"
        f"{args.pretrained_ds}_to_"
        f"{transfer_ds_alias}_"
        f"{args.model}_seed{args.seed}.json"
        f"initbeta{args.init_beta}_"
        f"betalr{args.beta_lr}.json"
    )

    if os.path.exists(result_path):
        print(
            f"Result already exists: {result_path}"
        )
        return

    f_name = get_file_name(__file__)

    log_file_path = set_logger(
        name=(
            f"{f_name}_"
            f"{args.pretrained_ds}_to_"
            f"{transfer_ds_alias}_"
            f"{args.model}_seed{args.seed}"
            f"betalr{args.beta_lr}"
        )
    )

    logger.info(
        f"Log file: {log_file_path}"
    )

    fix_seed(args.seed)
    
    wandb.init(
    project="stage-wise-ct",
    name=(
        f"stage_ct_{args.pretrained_ds}_to_"
        f"{transfer_ds_alias}_{args.model}_seed{args.seed}"
    ),
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

    # Load pretrained ResNet.
    model = get_pretrained_model(
        args.pretrained_ds,
        args.model,
    )

    # Freeze original backbone.
    for param in model.parameters():
        param.requires_grad = False

    # New Beans classifier remains trainable.
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

    # Replace ReLUs with our Stage-Wise CTUs.
    stage_model = (
        replace_resnet_relu_stagewise(
            copy.deepcopy(model),
            init_beta=args.init_beta,
            coeff=0.5,
        ).to(device)
    )

    initial_betas = get_stage_betas(
        stage_model
    )

    logger.info(
        f"Initial stage betas: "
        f"{initial_betas}"
    )

    curvature_params = sum(
        p.numel()
        for p in stage_model.stage_raw_betas.parameters()
    )

    total_trainable_params = sum(
        p.numel()
        for p in stage_model.parameters()
        if p.requires_grad
    )

    logger.info(
        f"Trainable curvature parameters: "
        f"{curvature_params}"
    )

    logger.info(
        f"Total trainable parameters: "
        f"{total_trainable_params}"
    )

    assert curvature_params == 4, (
        f"Expected exactly 4 beta parameters, "
        f"got {curvature_params}"
    )

    # Train.
    stage_model, best_val_acc = transfer(
        stage_model,
        train_loader,
        val_loader,
        beta_lr=args.beta_lr,
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

    logger.info(
        f"Stage-Wise CT Test Accuracy: "
        f"{test_acc:.2f}%"
    )

    logger.info(
        f"Final stage betas: "
        f"{final_betas}"
    )

    os.makedirs(
        "./ckpts",
        exist_ok=True,
    )

    checkpoint_path = (
        f"./ckpts/stage_ct_"
        f"{args.pretrained_ds}_to_"
        f"{transfer_ds_alias}_"
        f"{args.model}_seed{args.seed}.pth"
        f"betalr{args.beta_lr}.pth"
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
        initial_beta=args.init_beta,
        stage_betas=final_betas,
        coeff=0.5,
        best_val_acc=best_val_acc,
        beta_lr=args.beta_lr,
    )

    logger.info(
        f"Results saved to {result_path}"
    )

    wandb.finish()
    
if __name__ == "__main__":
    main()
