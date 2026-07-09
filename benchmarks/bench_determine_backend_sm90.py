import time
import torch
import json
import warnings
from unittest.mock import patch

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

import flashinfer.utils

def main():
    device = torch.device("cuda:0")
    
    # Verify CUDA through torch.cuda as requested
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in PyTorch")
        
    with patch("flashinfer.utils.get_compute_capability", return_value=(9, 0)):
        # Warmup
        for _ in range(200):
            flashinfer.utils.determine_attention_backend(
                device,
                0,
                False,
                False,
                torch.float16,
                torch.float16
            )
        
        # Measure
        iterations = 10000
        start = time.perf_counter_ns()
        for _ in range(iterations):
            flashinfer.utils.determine_attention_backend(
                device,
                0,
                False,
                False,
                torch.float16,
                torch.float16
            )
        end = time.perf_counter_ns()
        
        duration_ns = end - start
        ns_per_op = duration_ns / iterations
        
        # Output exact single JSON line for Perfloop
        print(json.dumps({"metric": "ns/op", "value": ns_per_op}))

if __name__ == "__main__":
    main()
