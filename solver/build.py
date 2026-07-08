from typing import Any, Dict, List

from utils.logger import log as logger

from utils.parse_module_str import parse_module_str


def build_optimizer(optimizer_cfg, model):
    """
    Build optimizer with different parameter groups.
    RULES:
        - Default learning rate is set for backbone
        - Other than backbone (newly initialized modules) will have different learning rate
            - Cross attention modules will have higher learning rate (> default)
            - Bias will have lower learning rate (> default)
            - classifier will have higher learning rate (> default)

    Args:
        optimizer_cfg: solver configuration
        model: model
    Returns:
        optimizer: optimizer
    """
    modifiable_optimizer_cfg = optimizer_cfg.copy()
    # Create parameter groups and set learning rate for each group

    param_group_configs = modifiable_optimizer_cfg.param_groups
    default_config = param_group_configs.pop("default", {})

    # Initialize parameter groups
    param_groups: List[Dict[str, Any]] = []
    assigned_params = set()

    # Create parameter groups based on config
    for group_name, group_config in param_group_configs.items():
        group_params = []

        # Iterate through all named parameters
        for name, param in model.named_parameters():
            # Skip if parameter does not require grad
            if not param.requires_grad:
                continue

            # Skip if parameter already assigned
            if id(param) in assigned_params:
                continue

            # Check if group name is partially included in parameter name
            if group_name in name:
                group_params.append(param)
                assigned_params.add(id(param))

        # Create parameter group if any parameters match
        if group_params:
            param_groups.append(
                {"params": group_params, "name": group_name, **group_config}
            )

    # Create default group for remaining parameters
    default_params = [
        param for param in model.parameters() if id(param) not in assigned_params
    ]
    if default_params:
        param_groups.append(
            {"params": default_params, "name": "default", **default_config}
        )

    # Remove empty groups
    param_groups = [group for group in param_groups if group["params"]]

    # Create optimizer
    optimizer_type = modifiable_optimizer_cfg.pop("type")
    optimizer_cls = parse_module_str(optimizer_type)
    optimizer = optimizer_cls(param_groups)

    logger.info(f"Using {optimizer_type} optimizer with information: {optimizer}")
    return optimizer


def build_lr_scheduler(scheduler_cfg, optimizer):
    """Build learning rate scheduler.
    Args:
        scheduler_cfg: solver configuration
    Returns:
        scheduler: learning rate scheduler
    """
    modifiable_scheduler_cfg = scheduler_cfg.copy()
    scheduler_type = modifiable_scheduler_cfg.pop("type")
    logger.info(f"Using {scheduler_type} scheduler.")
    lr_scheduler = parse_module_str(scheduler_type)(
        optimizer, **modifiable_scheduler_cfg
    )
    return lr_scheduler
