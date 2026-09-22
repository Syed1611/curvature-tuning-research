"""
This file implements the CTU (Curvature Tuning Unit) for CT and Trainable CTU.
It also provides utility functions to replace ReLU with CTU in a model and to compute the mean of beta and coeff parameters.
"""
import torch
from torch import nn
import torch.nn.functional as F


class SCTU(nn.Module):
    """
    CTU for Steering CT.
    """
    def __init__(self, shared_raw_beta, shared_raw_coeff, threshold=20):
        super().__init__()
        self.threshold = threshold
        self._raw_beta = shared_raw_beta
        self._raw_coeff = shared_raw_coeff
        self._raw_beta.requires_grad = False
        self._raw_coeff.requires_grad = False

    @property
    def beta(self):
        return torch.sigmoid(self._raw_beta)

    @property
    def coeff(self):
        return torch.sigmoid(self._raw_coeff)

    def forward(self, x):
        beta = torch.sigmoid(self._raw_beta)
        coeff = torch.sigmoid(self._raw_coeff)
        one_minus_beta = 1 - beta + 1e-6
        x_scaled = x / one_minus_beta

        return (coeff * torch.sigmoid(beta * x_scaled) * x +
                (1 - coeff) * F.softplus(x_scaled, threshold=self.threshold) * one_minus_beta)


class SWCTUFull(nn.Module):
    """
    Stage-Wise Curvature Tuning Unit with trainable beta and coeff.

    Each ResNet stage shares:
        - one trainable beta
        - one trainable coeff c
    """

    def __init__(
        self,
        shared_raw_beta,
        shared_raw_coeff,
        threshold=20
    ):
        super().__init__()

        self.threshold = threshold

        # Shared trainable parameters for this stage.
        self._raw_beta = shared_raw_beta
        self._raw_coeff = shared_raw_coeff

    @property
    def beta(self):
        return torch.sigmoid(self._raw_beta)

    @property
    def coeff(self):
        return torch.sigmoid(self._raw_coeff)

    def forward(self, x):
        beta = torch.sigmoid(self._raw_beta)
        coeff = torch.sigmoid(self._raw_coeff)

        one_minus_beta = 1 - beta + 1e-6
        x_scaled = x / one_minus_beta

        return (
            coeff
            * torch.sigmoid(beta * x_scaled)
            * x
            +
            (1 - coeff)
            * F.softplus(
                x_scaled,
                threshold=self.threshold
            )
            * one_minus_beta
        )


class TCTU(nn.Module):
    """
    CTU for Trainable CT.
    """
    def __init__(self, num_input_dims, out_channels, raw_beta=1.386, raw_coeff=0.0, threshold=20):
        super().__init__()
        self.threshold = threshold

        # Decide channel dim based on input shape
        if num_input_dims == 2 or num_input_dims == 3:  # (B, C) or (B, L, D)
            channel_dim = -1
        elif num_input_dims == 4: # (B, C, H, W)
            channel_dim = 1
        else:
            raise NotImplementedError(f"Unsupported input dimension {num_input_dims}")

        param_shape = [1] * num_input_dims
        param_shape[channel_dim] = out_channels

        # Init beta
        self._raw_beta = nn.Parameter(torch.full(param_shape, float(raw_beta)))

        # Init coeff
        self._raw_coeff = nn.Parameter(torch.full(param_shape, float(raw_coeff)))

    @property
    def beta(self):
        return torch.sigmoid(self._raw_beta)

    @property
    def coeff(self):
        return torch.sigmoid(self._raw_coeff)

    def forward(self, x):
        beta = torch.sigmoid(self._raw_beta)
        coeff = torch.sigmoid(self._raw_coeff)
        one_minus_beta = 1 - beta + 1e-6
        x_scaled = x / one_minus_beta

        return (coeff * torch.sigmoid(beta * x_scaled) * x +
                (1 - coeff) * F.softplus(x_scaled, threshold=self.threshold) * one_minus_beta)

class SWCTU(nn.Module):
    """
    Stage-Wise Curvature Tuning Unit.

    All CTUs belonging to the same ResNet stage share one
    trainable beta. The coefficient c remains fixed.
    """
    def __init__(self, shared_raw_beta, coeff=0.5, threshold=20):
        super().__init__()

        self.threshold = threshold

        # Shared trainable beta for this ResNet stage
        self._raw_beta = shared_raw_beta

        # c is fixed, not trainable
        self.register_buffer(
            "_coeff",
            torch.tensor(float(coeff), dtype=torch.float32)
        )

    @property
    def beta(self):
        return torch.sigmoid(self._raw_beta)

    @property
    def coeff(self):
        return self._coeff

    def forward(self, x):
        beta = torch.sigmoid(self._raw_beta)
        coeff = self._coeff

        one_minus_beta = 1 - beta + 1e-6
        x_scaled = x / one_minus_beta

        return (
            coeff * torch.sigmoid(beta * x_scaled) * x
            +
            (1 - coeff)
            * F.softplus(x_scaled, threshold=self.threshold)
            * one_minus_beta
        )

    
def replace_module(model, old_module=nn.ReLU, new_module=SCTU, **kwargs):
    """
    Replace all instances of old_module in the model with new_module.
    """
    device = next(model.parameters(), torch.tensor([])).device  # Handle models with no parameters

    # Replace modules
    for name, module in model.named_modules():
        if isinstance(module, old_module):
            ct = new_module(**kwargs).to(device)

            # Replace module in the model
            names = name.split(".")
            parent = model
            for n in names[:-1]:
                if n.isdigit():
                    parent = parent[int(n)]  # for Sequential/ModuleList
                else:
                    parent = getattr(parent, n)

            last_name = names[-1]
            if last_name.isdigit():
                parent[int(last_name)] = ct  # for Sequential/ModuleList
            else:
                setattr(parent, last_name, ct)

    return model


def replace_module_dynamic(model, input_shape, old_module=nn.ReLU, new_module=TCTU, **kwargs):
    """
    Replace all instances of old_module in the model with new_module that is dynamically created based on the number of output channels.
    """
    device = next(model.parameters(), torch.tensor([])).device
    dummy_input = torch.randn(*input_shape).to(device)

    module_metadata = {}  # name -> (num_input_dims, out_channels)
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            num_input_dims = input[0].dim()
            if num_input_dims in (2, 3):    # (B, C) or (B, L, D)
                out_channels = output.shape[-1]
            elif num_input_dims == 4:       # (B, C, H, W)
                out_channels = output.shape[1]
            else:
                raise NotImplementedError(f"Unsupported output shape {output.shape} in {name}")
            module_metadata[name] = (num_input_dims, out_channels)

        return hook

    # Register hooks to all modules of the target type
    for name, module in model.named_modules():
        if isinstance(module, old_module):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run dummy forward pass
    model(dummy_input)

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    # Replace modules
    for name, module in model.named_modules():
        if isinstance(module, old_module) and name in module_metadata:
            num_input_dims, out_channels = module_metadata[name]
            ct = new_module(num_input_dims=num_input_dims, out_channels=out_channels, **kwargs).to(device)

            # Replace module in the model
            names = name.split(".")
            parent = model
            for n in names[:-1]:
                if n.isdigit():
                    parent = parent[int(n)]  # for Sequential/ModuleList
                else:
                    parent = getattr(parent, n)

            last_name = names[-1]
            if last_name.isdigit():
                parent[int(last_name)] = ct  # for Sequential/ModuleList
            else:
                setattr(parent, last_name, ct)

    return model


def get_mean_beta_and_coeff(model):
    """
    Iterate through the model to compute the mean of beta and coeff parameters of TrainableCTU modules.
    """
    beta_vals = []
    coeff_vals = []

    for module in model.modules():
        if isinstance(module, TCTU):
            beta = module.beta.detach().flatten()
            coeff = module.coeff.detach().flatten()
            assert 0 <= beta.min() <= beta.max() <= 1
            assert 0 <= coeff.min() <= coeff.max() <= 1
            beta_vals.append(beta)
            coeff_vals.append(coeff)

    if beta_vals and coeff_vals:
        all_beta = torch.cat(beta_vals)
        all_coeff = torch.cat(coeff_vals)
        mean_beta = all_beta.mean().item()
        mean_coeff = all_coeff.mean().item()
        return mean_beta, mean_coeff
    else:
        return None, None


def replace_resnet_relu_stagewise(
    model,
    init_beta=0.8,
    coeff=0.5
):
    """
    Replace ResNet ReLUs with Stage-Wise CTUs.

    Stage mapping:
        Stage 1: stem ReLU + layer1
        Stage 2: layer2
        Stage 3: layer3
        Stage 4: layer4
    """

    if not (0.0 < init_beta < 1.0):
        raise ValueError("init_beta must be strictly between 0 and 1.")

    device = next(model.parameters()).device

    # Convert beta to the raw value used before sigmoid.
    raw_init = torch.logit(
        torch.tensor(init_beta, dtype=torch.float32, device=device)
    )

    # Exactly four trainable beta parameters.
    model.stage_raw_betas = nn.ParameterList([
        nn.Parameter(raw_init.clone())
        for _ in range(4)
    ])

    # Save the original ReLU names before replacing them.
    relu_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.ReLU)
    ]

    stage_counts = [0, 0, 0, 0]

    for name in relu_names:

        # ResNet stem + layer1 use beta_1
        if name == "relu" or name.startswith("layer1."):
            stage_idx = 0

        elif name.startswith("layer2."):
            stage_idx = 1

        elif name.startswith("layer3."):
            stage_idx = 2

        elif name.startswith("layer4."):
            stage_idx = 3

        else:
            raise ValueError(
                f"Unexpected ReLU location in ResNet: {name}"
            )

        stage_counts[stage_idx] += 1

        ct = SWCTU(
            shared_raw_beta=model.stage_raw_betas[stage_idx],
            coeff=coeff
        ).to(device)

        # Replace the original ReLU module.
        names = name.split(".")
        parent = model

        for n in names[:-1]:
            if n.isdigit():
                parent = parent[int(n)]
            else:
                parent = getattr(parent, n)

        last_name = names[-1]

        if last_name.isdigit():
            parent[int(last_name)] = ct
        else:
            setattr(parent, last_name, ct)

    print("Stage-Wise CT ReLU counts:", stage_counts)

    return model


def get_stage_betas(model):
    """
    Return the learned beta value for each ResNet stage.
    """
    return [
        torch.sigmoid(raw_beta).detach().item()
        for raw_beta in model.stage_raw_betas
    ]

def replace_resnet_relu_stagewise_full(
    model,
    init_beta=0.78,
    init_coeff=0.5
):
    """
    Replace ResNet ReLUs with 8-parameter Stage-Wise CTUs.

    Four trainable beta parameters + four trainable coeff parameters.

    Stage mapping:
        Stage 1: stem ReLU + layer1
        Stage 2: layer2
        Stage 3: layer3
        Stage 4: layer4
    """

    if not (0.0 < init_beta < 1.0):
        raise ValueError(
            "init_beta must be strictly between 0 and 1."
        )

    if not (0.0 < init_coeff < 1.0):
        raise ValueError(
            "init_coeff must be strictly between 0 and 1."
        )

    device = next(model.parameters()).device

    raw_beta_init = torch.logit(
        torch.tensor(
            init_beta,
            dtype=torch.float32,
            device=device
        )
    )

    raw_coeff_init = torch.logit(
        torch.tensor(
            init_coeff,
            dtype=torch.float32,
            device=device
        )
    )

    # Four shared beta parameters.
    model.stage_raw_betas = nn.ParameterList([
        nn.Parameter(raw_beta_init.clone())
        for _ in range(4)
    ])

    # Four shared coefficient parameters.
    model.stage_raw_coeffs = nn.ParameterList([
        nn.Parameter(raw_coeff_init.clone())
        for _ in range(4)
    ])

    # Save ReLU names before replacing them.
    relu_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.ReLU)
    ]

    stage_counts = [0, 0, 0, 0]

    for name in relu_names:

        if name == "relu" or name.startswith("layer1."):
            stage_idx = 0

        elif name.startswith("layer2."):
            stage_idx = 1

        elif name.startswith("layer3."):
            stage_idx = 2

        elif name.startswith("layer4."):
            stage_idx = 3

        else:
            raise ValueError(
                f"Unexpected ReLU location in ResNet: {name}"
            )

        stage_counts[stage_idx] += 1

        ct = SWCTUFull(
            shared_raw_beta=
                model.stage_raw_betas[stage_idx],
            shared_raw_coeff=
                model.stage_raw_coeffs[stage_idx],
        ).to(device)

        names = name.split(".")
        parent = model

        for n in names[:-1]:
            if n.isdigit():
                parent = parent[int(n)]
            else:
                parent = getattr(parent, n)

        last_name = names[-1]

        if last_name.isdigit():
            parent[int(last_name)] = ct
        else:
            setattr(parent, last_name, ct)

    print(
        "Stage-Wise Full CT ReLU counts:",
        stage_counts
    )

    return model


def get_stage_coeffs(model):
    """
    Return learned coefficient c for each ResNet stage.
    """

    return [
        torch.sigmoid(raw_coeff).detach().item()
        for raw_coeff in model.stage_raw_coeffs
    ]