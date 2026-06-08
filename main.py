import torch

import triton
import triton.language as tl
from triton.runtime import driver


DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def self_attn_fwd():
    pass


