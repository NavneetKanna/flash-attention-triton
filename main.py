import torch

import triton
import triton.language as tl
from triton.runtime import driver


DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def self_attn_fwd(
    Q, K, V, # (B, H, N, D)
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    scale,
    B, H, N, D,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_row = tl.program_id(0)
    batch_head_idx = tl.program_id(1)

    offset = batch_head_idx * stride_q_h

    # Load the blocks from VRAM to SRAM
    q_block_ptr = tl.make_block_ptr(
        base=Q+offset,
        shape=(N, BLOCK_D),
        strides=(stride_q_n, stride_q_d),
        offsets=(block_row*BLOCK_Q, 0),
        block_shape=(BLOCK_Q, BLOCK_D),
        order=(1, 0) # row major
    )

    # the shape is transposed
    k_block_ptr = tl.make_block_ptr(
        base=K+offset,
        shape=(BLOCK_D, N),
        strides=(stride_k_d, stride_k_n),
        offsets=(0, 0),
        block_shape=(BLOCK_D, BLOCK_KV),
        order=(0, 1) # column major
    )

    v_block_ptr = tl.make_block_ptr(
        base=V+offset,
        shape=(N, BLOCK_D),
        strides=(stride_v_n, stride_v_d),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, BLOCK_D),
        order=(1, 0) # row major
    )

    # variables for online softmax
    # these need to be 1D since we process in batch 
    # they apply across the rows of the block
    mi = tl.zeros([BLOCK_Q], dtype=tl.float32) - float('inf') # running max
    li = tl.zeros([BLOCK_Q], dtype=tl.float32)                # running denominator 
    o_acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)   # running output accumulator

    q_ptr = tl.load(q_block_ptr, boundary_check=(0, 1))

    for start_kv in range(0, N, BLOCK_KV):
        k_ptr = tl.load(k_block_ptr, boundary_check=(0, 1))
        v_ptr = tl.load(v_block_ptr, boundary_check=(0, 1))

        # Q @ K.T
        qk = tl.dot(q_ptr, k_ptr) * scale

        # Online softmax
        new_mi = tl.maximum(mi, tl.max(qk, axis=1)) # 1d
        alpha = tl.math.exp2(mi - new_mi) # 1d
        p = tl.math.exp2(qk - new_mi[:, None]) # 2d

        # Matmul with V
        o_acc = o_acc * alpha[:, None] + tl.dot(p, v_ptr)

        mi = new_mi
        li = li * alpha + tl.sum(p, axis=1) # 1d

        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_KV))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_KV, 0))

    o_acc = o_acc / li[:, None]

"""

Let's understand using an example:

Assume Q, K, V shapes (2, 2, 8, 4)

tensor([[[[0.3581, 0.1616, 0.5714, 0.4795],
          [0.5468, 0.3008, 0.9154, 0.3457],
          [0.4201, 0.1406, 0.2273, 0.5269],
          [0.1441, 0.1024, 0.8580, 0.8310],
          [0.7828, 0.5347, 0.0038, 0.2535],
          [0.3112, 0.3961, 0.2596, 0.3704],
          [0.7789, 0.6267, 0.0297, 0.9068],
          [0.9708, 0.1654, 0.0144, 0.4128]],

         [[0.7147, 0.2785, 0.6463, 0.7070],
          [0.0046, 0.2647, 0.3889, 0.6876],
          [0.3702, 0.3406, 0.5874, 0.0967],
          [0.2227, 0.3751, 0.3261, 0.0857],
          [0.1634, 0.6659, 0.6811, 0.6651],
          [0.7258, 0.4927, 0.7543, 0.2057],
          [0.1071, 0.8613, 0.2727, 0.3571],
          [0.1453, 0.2662, 0.1778, 0.9726]]],


        [[[0.6436, 0.7744, 0.0494, 0.0897],
          [0.8326, 0.0759, 0.1208, 0.3943],
          [0.5721, 0.9949, 0.4025, 0.8175],
          [0.3231, 0.4774, 0.9158, 0.3784],
          [0.9886, 0.4412, 0.2792, 0.2915],
          [0.7545, 0.5258, 0.3754, 0.5061],
          [0.1726, 0.5226, 0.8953, 0.3112],
          [0.2522, 0.9481, 0.3493, 0.3176]],

         [[0.9558, 0.9222, 0.7712, 0.1684],
          [0.7397, 0.6067, 0.9695, 0.3222],
          [0.6258, 0.9842, 0.9909, 0.3784],
          [0.1019, 0.5079, 0.9599, 0.1936],
          [0.7737, 0.6805, 0.4499, 0.2204],
          [0.2879, 0.6809, 0.8073, 0.9478],
          [0.4164, 0.1058, 0.6489, 0.4592],
          [0.4539, 0.3612, 0.0810, 0.8282]]]])

Assuming:
 BLOCK_Q = 4
 BLOCK_D = 4
 stride_q_h = 32

Now the grid is launched as (8/BLOCK_Q, 2*2) = (2, 4)

So the grid is

[block 0 (0, 0), block 1 (0, 1), (0, 2), (0, 3)
 block 4 (1, 0), (1, 1), (1, 2), (1, 3)]

So now lets see what happens inside block 0 when we execute the kernel

block_row = 0
batch_head_idx = 0

offset = 0 * 32 = 0

q_block_ptr = tl.make_block_ptr(
    base=Q+offset,
    shape=(N, BLOCK_D),
    strides=(stride_q_n, stride_q_d),
    offsets=(block_row*BLOCK_Q, 0),
    block_shape=(BLOCK_Q, BLOCK_D),
    order=(1, 0) # row major
)

So what is happening here is that, we are making a block pointer and telling it to start at position (0, 0, 0, 0). This is because the offset
is 0 and Q points to the starting element, so in the base arg we tell triton that the tensor starts at (0, 0, 0, 0). Next for the shape arg
we tell triton that the shape of the tensor is (8, 4), so its basically a view. Now for the offseats arg we need to calculate it based on the
block id, because if you observe we have launched the grid by dividing 8/4, so which means, for block 0, the offsets will be 0*4 = 0, (0, 0).
This tells triton in the tensor starting at (0, 0, 0, 0) which has shape (8, 4) start the pointer at postion (0, 0) and this block has shape
(4, 4). Similarly, block 4 would have base=(0, 0, 0, 0), shape=(8, 4), offsets as (4, 0), etc.

To visualize for the Q matrix:

Block 0 would load: base = (0, 0, 0, 0); shape = (8, 4); offsets = (0, 0); block_shape = (4, 4)
    [0.3581, 0.1616, 0.5714, 0.4795]
    [0.5468, 0.3008, 0.9154, 0.3457]
    [0.4201, 0.1406, 0.2273, 0.5269]
    [0.1441, 0.1024, 0.8580, 0.8310]

Block 1 would load: base = (0, 0, 8, 4); shape = (8, 4); offsets = (0, 0); block_shape = (4, 4)
    [0.7147, 0.2785, 0.6463, 0.7070]
    [0.0046, 0.2647, 0.3889, 0.6876]
    [0.3702, 0.3406, 0.5874, 0.0967]
    [0.2227, 0.3751, 0.3261, 0.0857]

Block 4 would load: base = (0, 0, 0, 0); shape = (8, 4); offsets = (4, 0); block_shape = (4, 4)
    [0.7828, 0.5347, 0.0038, 0.2535]
    [0.3112, 0.3961, 0.2596, 0.3704]
    [0.7789, 0.6267, 0.0297, 0.9068]
    [0.9708, 0.1654, 0.0144, 0.4128]

etc

So, we load a block from Q to SRAM. Similarly, we load a block from K and V, but we do it inside a strided loop (Block_KV). Unlike Q, the offsets
arg will be 0 since we will be streaming through them in the loop.

So to summarize:

- BLOCK_Q = 4, BLOCK_D = 4, BLOCK_KV = 4
- stride_q_h = 32
- Q, K, V has shape (2, 2, 8, 4) where 2 is bs, 2 is no of heads, 8 is seq len, 4 is head dim
- Grid is launched as (N/BLOCK_Q, 2*2) = (2, 4)
- For each block in the grid we load these blocks for each matrix:
 - block_row = tl.program_id(0)
 - batch_head_idx = tl.program_id(1)
 - offset = batch_head_idx * stride_q_h
 - Q_block:
    - base is Q+offset
    - shape is (N, BLOCK_D) = (8, 4)
    - offsets is (block_row*BLOCK_Q, 0)
    - block_shape is (BLOCK_Q, BLOCK_D) = (4, 4)
 - K_block (note we loaded it transposed):
   - base is K+offset
   - shape is (BLOCK_D, N) = (4, 8)
   - offsets is (0, 0)
   - block_shape is (BLOCK_D, BLOCK_KV) = (4, 4)
 - V_block:
   - base is V+offset
   - shape is (N, BLOCK_D) = (8, 4)
   - offsets is (0, 0)
   - block_shape is (BLOCK_KV, BLOCK_D) = (4, 4)

Taking block 0 as example, and assuming Q, K, V are the same tensor, it is loading into SRAM

    [0.3581, 0.1616, 0.5714, 0.4795]
    [0.5468, 0.3008, 0.9154, 0.3457]
    [0.4201, 0.1406, 0.2273, 0.5269]
    [0.1441, 0.1024, 0.8580, 0.8310]

Now K.T looks like this

    [0.3726, 0.2675, 0.6271, 0.0074, 0.7765, 0.4731, 0.5758, 0.5959]
    [0.9796, 0.5407, 0.0052, 0.7666, 0.0496, 0.4861, 0.4594, 0.3132]
    [0.9050, 0.2445, 0.9567, 0.0332, 0.0430, 0.3592, 0.2487, 0.9765]
    [0.7481, 0.3501, 0.0109, 0.0422, 0.3608, 0.8837, 0.6314, 0.7293]

So we load (4, 4) block
    [0.3726, 0.2675, 0.6271, 0.0074]
    [0.9796, 0.5407, 0.0052, 0.7666]
    [0.9050, 0.2445, 0.9567, 0.0332]
    [0.7481, 0.3501, 0.0109, 0.0422]

and perform dot product between q block and k block. After that, we do online softmax and dot product with V.

    [0.3581, 0.1616, 0.5714, 0.4795]
    [0.5468, 0.3008, 0.9154, 0.3457]
    [0.4201, 0.1406, 0.2273, 0.5269]
    [0.1441, 0.1024, 0.8580, 0.8310]

next iteration, we advance k block right and advance v block down.

"""
