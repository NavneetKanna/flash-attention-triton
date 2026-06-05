import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # If vector len = 256 and block size = 64
    # there would be 4 programs (pid - 0, 1, 2, 3) = [0:64, 64:128, 128:192, 192:256]
    block_start = pid * BLOCK_SIZE
    indices = block_start + tl.arange(0, BLOCK_SIZE)

    mask = indices < n_elements

    x = tl.load(x_ptr + indices, mask=mask)
    y = tl.load(y_ptr + indices, mask=mask)

    out = x + y

    tl.store(out_ptr + indices, out, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    pass

def main():

     
