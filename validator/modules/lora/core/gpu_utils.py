try:
    from torch.cuda import get_device_name
except (ImportError, RuntimeError):
    # torch not compiled with CUDA or unavailable
    get_device_name = None


def get_gpu_type() -> str:
    """Return a descriptive GPU string or an error message if unavailable."""
    if get_device_name is None:
        return "CPU/Unknown"
    try:
        return get_device_name(0)
    except Exception as e:
        return f"Error retrieving GPU type: {e}" 