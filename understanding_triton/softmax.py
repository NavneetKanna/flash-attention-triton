import torch

import triton
import triton.language as tl
from triton.runtime import driver

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





