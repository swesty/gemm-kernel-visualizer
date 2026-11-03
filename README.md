# GEMM kernels 1–7 + Tensor Cores: thread view + memory view

An interactive, single-file visualizer for nine GEMM kernels, from a naive kernel up to Tensor Core WMMA. It steps through one block and one warp at a time and shows which elements of A, B, C and the shared-memory tiles each lane touches.

**Open it:** https://swesty.github.io/gemm-kernel-visualizer/ (or download `index.html` and open it locally; it has no dependencies).

## Kernels

**Lecture 8 / Week 2-4** (fp32, M = 1000, K = 1200, N = 800)


1. **Naive, x → row.** A warp spans a column of C. A loads hit 32 sectors per instruction; B is a broadcast.
2. **Naive, x → col.** A warp spans a row of C. A is a broadcast; B loads hit 4 sectors per instruction.
3. **Shared-memory tiling (32×32).** A cooperative load phase, `syncthreads`, then a compute phase with 2 shared-memory reads per FMA.
4. **1D block tiling (BM = BN = BK = 32, TM = 8, 128 threads).** A strided cooperative load (`idx = tid + 128·j`), and each `b_val` reused across 8 accumulators (1.125 shared reads per FMA).

**Lecture 10** (fp32, M = N = K = 1024)

5. **2D block tiling.** 64×64 block, 256 threads, each owning a 4×4 patch. The outer product gives 0.5 shared reads per FMA, and the bank model shows 2-way conflicts on the compute reads.
6. **Warp tiling.** Adds a warp tier: 16×32 warp regions, each holding 32 lane patches. tileA is stored transposed and loads move groups of 4 floats. The compute reads become conflict-free.
7. **Kernel factory.** Pick any of the 11 `(BM, BN, BK, WM, WN, TM, TN)` candidates from matmul-07. The page shows the validator result, thread count, shared bytes, idle load threads and reads per FMA.

**Lecture 12** (fp16 in, fp32 accumulate, WMMA 16×16×16)

- **TC-01.** One warp computes one tile with collective calls. B is declared col_major, so its sectors run down columns.
- **TC-02.** Any size by padding: one 16×16 tile per warp, 4 warps per block, K padded to 1008 (63 chunks). It includes the uniform warp exit at `blockIdx.y = 18`.

WMMA fragments are drawn per warp, never per lane, because the portable API does not specify which lane holds which element.

## What it shows

- Panels for C (thread ownership), A, B, tileA and tileB. A gridline every 8 columns marks a 32-byte sector.
- A registers panel with one column per lane: the accumulators, plus `b_val` for kernel 4, `regM`/`regN` for kernels 5–7, and the fragments for the Tensor Core kernels.
- A "show all warps" toggle on the load steps.
- Per-step metrics: global sectors per instruction, shared-memory wavefronts, and reads per FMA.
- The source line for the current step, highlighted.
- A lane inspector.

Coalescing is analyzed per warp at a single k-step, which is how the hardware sees it, not along one thread's loop.
