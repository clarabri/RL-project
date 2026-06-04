import torch


def resolve_device(accelerator: str, devices: list[int]) -> torch.device:
    """Resolve a Lightning-style accelerator + devices config to a torch.device.

    Args:
        accelerator: "cpu", "gpu", or "mps"
        devices: list of device indices (used when accelerator="gpu")

    Returns:
        torch.device ready for use

    Raises:
        ValueError: unknown accelerator string
        RuntimeError: requested hardware is unavailable
    """
    accelerator = accelerator.lower()

    if accelerator == "cpu":
        return torch.device("cpu")

    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "accelerator='gpu' requested but CUDA is not available. "
                "Use accelerator='cpu' or check your CUDA installation."
            )
        if not devices:
            raise ValueError("accelerator='gpu' requires at least one device index in devices=[...]")
        idx = devices[0]
        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"GPU device index {idx} requested but only "
                f"{torch.cuda.device_count()} GPU(s) available."
            )
        return torch.device(f"cuda:{idx}")

    if accelerator == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "accelerator='mps' requested but MPS is not available on this machine."
            )
        return torch.device("mps")

    raise ValueError(
        f"Unknown accelerator '{accelerator}'. Choose from: 'cpu', 'gpu', 'mps'."
    )
