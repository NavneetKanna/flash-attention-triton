import torch

import triton
import triton.language as tl
from triton.runtime import driver


"""
The device I will be running on is rtx 4090. These are some of the specs:

1. Warp size = 32
2. Block = Max 1024 threads or 32 warps
3. No of SM = 128
4. Theoritically 16 blocks can run in 1 SM simultaneously. But usually, this is not achieved since different warps utilize
   different no of registers, shared memory, etc, which limits the no of blocks that can run in 1 SM simultaneously.
5. No of registers = 65536
6. 1 SM can run at max 32 blocks
7. 16384 cuda cores, 128 per SM
8. 512 tensor cores, 4 per SM
9. 1 SM has 128kb of SRAM memory, but 100kb is given to programmers. Acts like a L1 cache, and is unique to each SM
10. 72MB of L2 cache, this is a global cache across all SM's
"""


DEVICE = triton.runtime.driver.active.get_active_torch_device()

def naive_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    exp_x = torch.exp(x - x_max)

    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x

@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_rows, row_stride, BLOCK_SIZE: tl.constexpr, n_cols):
    row_start = tl.program_id(0)
    row_offset = tl.num_programs(0) # 768

    # there are 768 programs launched
    # row_start will be 0..768
    # and we loop in grid stride loop, for example,
    # assuming we are row 5, the loop indicies will be 5, 5+768 < 1873, 5+768+768 < 1873 
    for row_idx in tl.range(row_start, n_rows, row_offset):
        row_start_ptr = x_ptr + row_idx*row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        x_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols

        # load the row to sram
        # assuming fp32 and 1024 cols, 1 row = 4kb
        row = tl.load(x_ptrs, mask=mask, other=-float('inf'))

        row_minus_max = row - tl.max(row, axis=0)
        num = tl.exp(row_minus_max)
        dem = tl.sum(num, axis=0)
        softmax_output = num / dem

        # write output back to vram
        output_row_start_ptr = out_ptr + row_idx*row_stride
        out_ptrs = output_row_start_ptr + col_offsets
        tl.store(out_ptrs, softmax_output, mask=mask)

# {'max_shared_mem': 101376, 'max_num_regs': 65536, 'multiprocessor_count': 128, 'warpSize': 32, 'sm_clock_rate': 2520000, 'mem_clock_rate': 10501000, 'mem_bus_width': 384}
properties = driver.active.utils.get_device_properties(DEVICE.index)

# 128
NUM_SM = properties["multiprocessor_count"]

# 65536
NUM_REGS = properties["max_num_regs"]

# 101376 bytes ~ 100kb
SIZE_SMEM = properties["max_shared_mem"]

# 32
WARP_SIZE = properties["warpSize"]

def softmax(x):
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # by specifying this, we set threads_per_block
    # 8*32 = 256 threads per block
    num_warps = 8

    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_warps=num_warps, grid=(1,))
    kernel._init_handles()
    # this specifies how many registers are used by 1 thread
    # for this example, i get 37
    n_regs = kernel.n_regs
    # this tells us how much sram in bytes is used by 1 block
    # for this example, i get 4kb
    size_smem = kernel.metadata.shared

    # the denominator gives us how many registers are used by 1 block
    # for this example, it gives 6
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)

    # we choose the min between how many registers 1 blocks uses
    # vs sram it uses
    # for this example, it is min(6, 24)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)

    # this gives us the number of blocks that can run across
    # all sm's
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)

    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE)
    return y

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=DEVICE)
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


