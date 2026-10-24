# Publication CUDA run

`h200-publication.json` contains measurements of this repository’s `minidynamo` package, including every timing trial and SHA-256 digests of the package modules and benchmark script. The digests identify the exact measured files independently of Git history.

The final run used SSH to a Cartesia login node and an interactive Slurm allocation: one NVIDIA H200, four CPU cores, and 32 GiB host memory. See the JSON for Python, PyTorch, CUDA, driver, device, timestamp, and Slurm job ID.

```bash
uv sync --frozen --group dev
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
uv run pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
uv run python examples/benchmark_publication.py --warmup 500 --iterations 1000 --repeats 15 \
  --output benchmarks/h200-publication.json
```

Validation: **244 passed, 4 skipped**. The skips are the four MPS tests on a CUDA node. Tests cover CUDA float32/float16/bfloat16 results, Python/JIT gradients, non-contiguous inputs, metadata/context guards, and CPU graph/kernel parity. CUDA tests compare complete ATen graphs and nonempty Triton source for four programs at 32×32 and the longer chain at 2048×2048. Both capture paths use the same `make_fx` and `compile_fx_inner` calls. `h200-inductor-evidence.json` saves the large graph, both generated Triton kernels, comparison results, and source digests. This establishes parity through that shared inference path, not equivalence to the full `torch.compile` pipeline.

Each timing is the median of fifteen randomized-order trials, 1,000 calls per trial, after 500 warmups. Tensor dtype is float32, shapes are 32×32, 256×256, and 2048×2048, inference mode is enabled, and both Inductor paths have CUDA graphs disabled. Synchronization brackets each timing block. This measures amortized wall time including CPU dispatch, not isolated kernel latency or cold-start compilation. Correctness checks cover original and changed tensor contents at rtol=1e-4 and atol=1e-3. Small-input timings include host-side variation; inspect the full trials and ranges instead of interpreting tenths of a microsecond as stable differences.

Profiler collection happens after the timed trials. A warmup call initializes the activity collector; two subsequent marked calls must produce matching CUDA activity lists. Activities are assigned to calls using CPU/CUDA correlations. `cuda_events` includes correlated memory sets/copies; `kernel_names` and `kernel_count` exclude those. An inconsistent sample is reported as a profiler error, with no inferred count. All 2048×2048 samples agreed: eager/Python launched twelve compute kernels; TorchScript and both Inductor paths launched two. TorchScript fused the eleven elementwise operations and is not an unfused baseline.

`h200-patterns.txt` preserves a separate correctness and timing run of `examples/benchmark_transformers.py` on the same allocation. These are single timing blocks (50 warmups, 1,000 calls), not the repeated experiment above. The results include both speedups and slowdowns; consult the recorded table for each path. These patterns are not full transformer layers or per-token normalization implementations.

The published table measures a small chain plus reduction, not transformer-model performance or a general backend ranking. Raw and guarded callables have separate rows. MPS behavior and CUDA training through the direct private Inductor bridge are not validated; the latter is explicitly unsupported. The bridge also bypasses the standard functionalization/decomposition pipeline, so some tracer-supported operations can fail during Inductor lowering.
