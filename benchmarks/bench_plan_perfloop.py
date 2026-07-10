import numpy as np
import time
import torch
import flashinfer


def main():
    import build_backend

    build_backend._create_data_dir(use_symlinks=True)

    batch_size = 128
    seq_len = 1024
    num_qo_heads = 32
    num_kv_heads = 4
    head_dim = 128
    page_block_size = 16
    q_dtype = torch.bfloat16
    kv_dtype = torch.bfloat16

    np.random.seed(42)
    seq_lens = torch.full((batch_size,), seq_len)
    seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()
    kv_indptr = (
        torch.cat([torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0)
        .int()
        .to("cuda:0")
    )

    num_blocks = kv_indptr[-1].item()
    indices = torch.arange(num_blocks).int().to("cuda:0")
    last_page_len = (
        (seq_lens - (seq_lens_blocks - 1) * page_block_size).int().to("cuda:0")
    )

    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD", use_tensor_cores=True
    )

    # 1. Warm-up iterations to JIT compile kernels and stabilize CPU/GPU states
    for _ in range(20):
        wrapper.plan(
            kv_indptr,
            indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_block_size,
            data_type=kv_dtype,
            q_data_type=q_dtype,
        )

    # 2. Run statistical timed iterations using CPU wall-clock timer (time.perf_counter)
    durations = []
    for _ in range(100):
        start = time.perf_counter()
        wrapper.plan(
            kv_indptr,
            indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_block_size,
            data_type=kv_dtype,
            q_data_type=q_dtype,
        )
        end = time.perf_counter()
        durations.append((end - start) * 1000.0)  # convert to milliseconds

    # 3. Calculate median execution time to handle scheduling noise and CPU scheduling jitter
    median_ms = np.median(durations)

    # Emit exact Perfloop JSONL format
    print(f'{{"metric": "ms", "value": {median_ms:.6f}}}')


if __name__ == "__main__":
    main()
