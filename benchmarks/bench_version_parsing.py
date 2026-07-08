import sys
import time
import json
import functools
from unittest.mock import MagicMock
from packaging.version import Version as TorchVersion

# 1. Metaclasses for DummyTorch and cuda
class Metaclass(type):
    def __getattr__(cls, name):
        return MagicMock()

class CudaMetaclass(type):
    def __getattr__(cls, name):
        return MagicMock()

# Create custom lightweight torch
class DummyTorch(metaclass=Metaclass):
    class dtype:
        def __init__(self, name):
            self.name = name
        def __repr__(self):
            return self.name

    float16 = dtype("float16")
    bfloat16 = dtype("bfloat16")
    float32 = dtype("float32")
    float64 = dtype("float64")
    int8 = dtype("int8")
    int16 = dtype("int16")
    int32 = dtype("int32")
    int64 = dtype("int64")
    uint8 = dtype("uint8")
    uint32 = dtype("uint32")
    float8_e4m3fn = dtype("float8_e4m3fn")
    float8_e5m2 = dtype("float8_e5m2")

    class version:
        cuda = '12.4'

    class cuda(metaclass=CudaMetaclass):
        @staticmethod
        def is_available():
            return True
        @staticmethod
        def get_device_capability(index=0):
            return (9, 0)
        @staticmethod
        def get_device_properties(index=0):
            mock_prop = MagicMock()
            mock_prop.shared_memory_per_block_optin = 16384
            return mock_prop
        @staticmethod
        def device_count():
            return 1
    
    __version__ = '2.5.0'
    _C = MagicMock()
    _C._GLIBCXX_USE_CXX11_ABI = True
    torch_version = MagicMock()
    torch_version.__version__ = '2.5.0'
    torch_version.TorchVersion = TorchVersion

sys.modules['torch'] = DummyTorch
sys.modules['torch.nn'] = DummyTorch
sys.modules['torch.nn.functional'] = DummyTorch
sys.modules['torch.version'] = DummyTorch.version
sys.modules['torch.torch_version'] = DummyTorch.torch_version
sys.modules['torch.cuda'] = DummyTorch.cuda

sys.modules['pynvml'] = MagicMock()
sys.modules['tqdm'] = MagicMock()
sys.modules['requests'] = MagicMock()

tvm_ffi = MagicMock()
tvm_ffi.__version__ = '0.1.12'
tvm_ffi.libinfo = MagicMock()
tvm_ffi.libinfo.find_include_path.return_value = '/mock/include'
tvm_ffi.libinfo.find_dlpack_include_path.return_value = '/mock/dlpack'
sys.modules['tvm_ffi'] = tvm_ffi

filelock = MagicMock()
filelock.FileLock = MagicMock()
filelock.Timeout = Exception
sys.modules['filelock'] = filelock

jinja2 = MagicMock()
jinja2.Template = lambda text: MagicMock(render=lambda **kwargs: text)
sys.modules['jinja2'] = jinja2

sys.path.insert(0, ".")

# 2. Import functions
from flashinfer.utils import determine_attention_backend
import flashinfer.utils

# 3. Override get_compute_capability
flashinfer.utils.get_compute_capability = lambda device: (9, 0)

# Create dummy device
class DummyDevice:
    def __init__(self):
        self.type = "cuda"
        self.index = 0

device = DummyDevice()

# 4. Warm up
for _ in range(5000):
    determine_attention_backend(
        device,
        pos_encoding_mode=0,
        use_fp16_qk_reductions=False,
        use_custom_mask=False,
        dtype_q=DummyTorch.float16,
        dtype_kv=DummyTorch.float16,
    )

# 5. Measure
num_iterations = 50000
start_time = time.perf_counter()
for _ in range(num_iterations):
    determine_attention_backend(
        device,
        pos_encoding_mode=0,
        use_fp16_qk_reductions=False,
        use_custom_mask=False,
        dtype_q=DummyTorch.float16,
        dtype_kv=DummyTorch.float16,
    )
end_time = time.perf_counter()

total_duration_ns = (end_time - start_time) * 1e9
ns_per_op = total_duration_ns / num_iterations

# 6. Output JSON result
print(json.dumps({"metric": "ns/op", "value": ns_per_op}))
