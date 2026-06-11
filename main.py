import torch

import triton
import triton.language as tl
from triton.runtime import driver


DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def self_attn_fwd(
    Q, K, V, # (B, H, N, D)
    q_shape,
    k_shape,
    v_shape,
    q_stride,
    k_stride,
    v_stride,
    Q_BLOCK_SHAPE_ROW,
    Q_BLOCK_SHAPE_COL,
):
    """
    For example, let:

    - Q, K, V = (2, 4, 512, 64)
    - Q_BLOCK_SHAPE_ROW = 128
    - Q_BLOCK_SHAPE_COL = 64
    - Grid = (512/128, 2*4) = (4, 8)

    """

    # output shape is (B, H, N, D)

    # grid is launched as (B*H, N/BLOCK_M)

    q_block = tl.make_block_ptr(
        Q,
        q_shape,
        q_stride,
        offsets=(0, 0),
        block_shape=(Q_BLOCK_SHAPE_ROW, Q_BLOCK_SHAPE_COL),
        order=(1, 0) # row major
    )



