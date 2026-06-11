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
    K_BLOCK_SHAPE_ROW,
    K_BLOCK_SHAPE_COL,
    V_BLOCK_SHAPE_ROW,
    V_BLOCK_SHAPE_COL,
):
    """
    For example, let:

    - Q, K, V = (2, 4, 512, 64)
    - Q_BLOCK_SHAPE_ROW = 128
    - Q_BLOCK_SHAPE_COL = 64
    - Grid = (512/128, 2*4) = (4, 8)
     - 4 blocks along the sequence (N) dimension
     - 8 blocks for (batch x head) combinations
     - This grid launches 32 threadblocks that run this kernel
       independently

    """

    # output shape is (B, H, N, D)

    # Load the blocks from VRAM to SRAM
    q_block_ptr = tl.make_block_ptr(
        Q,
        q_shape,
        q_stride,
        offsets=(0, 0),
        block_shape=(Q_BLOCK_SHAPE_ROW, Q_BLOCK_SHAPE_COL),
        order=(1, 0) # row major
    )

    k_block_ptr = tl.make_block_ptr(
        K,
        k_shape,
        k_stride,
        offsets=(0, 0),
        block_shape=(K_BLOCK_SHAPE_ROW, K_BLOCK_SHAPE_COL),
        order=(1, 0) # row major
    )

    v_block_ptr = tl.make_block_ptr(
        V,
        v_shape,
        v_stride,
        offsets=(0, 0),
        block_shape=(V_BLOCK_SHAPE_ROW, V_BLOCK_SHAPE_COL),
        order=(1, 0) # row major
    )

    # We need not specify mask since triton takes care of it
    # when we use block ptr, but we can pass boundry check
    # which specifies the dims we want to check for illegal access
    q_block = tl.load(q_block_ptr, boundary_check=(0, 1))
    k_block = tl.load(k_block_ptr, boundary_check=(0, 1))
    v_block = tl.load(v_block_ptr, boundary_check=(0, 1))

    # With the blocks loaded, we can do all the steps for attn
    # in one go without storing the itermediate results back to VRAM

    # Step 1: transpose the last two dims of K
    tl.trans(k_block, (0, 1, 3, 2))

    #  Step 2: Q @ K.T

    # Here I am using K_BLOCK_SHAPE_ROW instead of K_BLOCK_SHAPE_COL
    # since K is now transposed
    acc = tl.zeros(Q_BLOCK_SHAPE_ROW, K_BLOCK_SHAPE_ROW)
    acc = tl.dot(q_block, k_block, acc)





