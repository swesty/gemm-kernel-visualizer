# GEMM kernels 1–4: thread view + memory view

An interactive, single-file visualizer for four fp32 GEMM kernels (M = 1000, K = 1200, N = 800, row-major). It steps through one block and one warp at a time and shows which elements of A, B, C and the shared-memory tiles each lane touches.

**Open it:** https://swesty.github.io/gemm-kernel-visualizer/ (or download `index.html` and open it locally; it has no dependencies).

## The four kernels

1. **Naive, x → row.** A warp spans a column of C. A loads hit 32 sectors per instruction; B is a broadcast.
2. **Naive, x → col.** A warp spans a row of C. A is a broadcast; B loads hit 4 sectors per instruction.
3. **Shared-memory tiling (32×32).** A cooperative load phase, `syncthreads`, then a compute phase with 2 shared-memory reads per FMA.
4. **1D block tiling (BM = BN = BK = 32, TM = 8, 128 threads).** A strided cooperative load (`idx = tid + 128·j`), and each `b_val` reused across 8 accumulators (1.125 shared reads per FMA).

## What it shows

- Panels for C (thread ownership), A, B, tileA and tileB. A gridline every 8 columns marks a 32-byte sector.
- A registers panel with one column per lane: the accumulators, plus `b_val` for kernel 4.
- A "show all warps" toggle on the load steps.
- Per-step metrics: global sectors per instruction, shared-memory wavefronts, and reads per FMA.
- The source line for the current step, highlighted.
- A lane inspector.

Coalescing is analyzed per warp at a single k-step, which is how the hardware sees it, not along one thread's loop.
