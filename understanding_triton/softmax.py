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
def softmax_kernel(x_ptr, out_ptr, n_rows):
    row_start = tl.program_id(0)
    row_offset = tl.num_programs(0) # 768

    # there are 768 programs launched
    # row_start will be 0..768
    # and we loop in grid stride loop, for example,
    # assuming we are row 5, the loop indicies will be 5, 5+768 < 1873, 5+768+768 < 1873 
    for row_idx in tl.range(row_start, n_rows, row_offset):

