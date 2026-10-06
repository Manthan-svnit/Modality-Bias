import torch
import logging

logger = logging.getLogger(__name__)


def get_device(requested_device: str = "auto") -> torch.device:
    """Select compute device automatically or according to user specification.

    Args:
        requested_device: 'auto', 'cuda', 'cpu', or specific device like 'cuda:0'.

    Returns:
        torch.device instance.
    """
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logger.info(f"Using GPU device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available. Using CPU.")
    else:
        device = torch.device(requested_device)
        logger.info(f"Using explicitly requested device: {device}")
    return device
