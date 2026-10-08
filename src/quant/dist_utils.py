import torch.distributed as dist


def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_available_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def is_main():
    return get_rank() == 0


def print_on_main(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)
