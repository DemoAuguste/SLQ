"""Backend device abstraction: NVIDIA CUDA or Ascend NPU.

SLQ 源码在 CUDA 与昇腾 NPU 上共用一套代码，所有设备/后端的判断都收口到这里。
NPU 由 torch_npu 提供；未安装 torch_npu 时回退 CUDA/CPU。
"""
import torch

_NPU = None


def _detect_npu() -> bool:
    global _NPU
    if _NPU is None:
        try:
            import torch_npu  # noqa: F401
            _NPU = bool(torch_npu.npu.is_available())
        except Exception:
            _NPU = False
    return _NPU


def is_npu() -> bool:
    """是否为 Ascend NPU 环境。"""
    return _detect_npu()


def device_type() -> str:
    """返回 'npu' / 'cuda' / 'cpu'。"""
    if is_npu():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_of(rank: int = 0) -> str:
    """返回 '<type>:<rank>' 设备字符串。"""
    return f"{device_type()}:{rank}"


def set_device(rank: int = 0) -> None:
    """将当前进程绑定到 rank 对应的物理设备。"""
    d = device_of(rank)
    if is_npu():
        import torch_npu
        torch_npu.npu.set_device(d)
    elif torch.cuda.is_available():
        torch.cuda.set_device(d)


def empty_cache() -> None:
    """释放设备缓存（CUDA / NPU 各自实现）。"""
    if is_npu():
        import torch_npu
        torch_npu.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_group_backend() -> str:
    """分布式后端：NPU 用 'hccl'，CUDA 用 'nccl'。"""
    return "hccl" if is_npu() else "nccl"


def safe_set_tf32(enabled: bool = False) -> None:
    """在支持的平台（仅 CUDA）上关闭/开启 TF32，保证 matmul 确定性；NPU/CPU 上无操作。"""
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = enabled
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = enabled
        except Exception:
            pass