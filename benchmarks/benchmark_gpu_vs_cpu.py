"""
GPU vs CPU Performance Benchmarks for sigdiscovPy.

Measures speedup of CuPy GPU implementations over NumPy CPU implementations.
"""

import time
from typing import Dict, List, Optional
import numpy as np

from sigdiscovpy.gpu.backend import GPU_AVAILABLE, get_array_module
from sigdiscovpy.core.normalization import standardize_matrix
from sigdiscovpy.core.weights import create_gaussian_weights
from sigdiscovpy.core.spatial_lag import compute_spatial_lag
from sigdiscovpy.core.metrics import compute_moran_from_lag, compute_ind_from_lag
from sigdiscovpy.analysis.pairwise_moran import pairwise_moran


def generate_test_data(n_genes: int, n_cells: int, seed: int = 42):
    """Generate random test data for benchmarking."""
    rng = np.random.default_rng(seed)

    # Expression matrix (genes x cells)
    expr = rng.lognormal(0, 1, (n_genes, n_cells)).astype(np.float32)

    # Random 2D coordinates
    coords = rng.uniform(0, 1000, (n_cells, 2)).astype(np.float32)

    return expr, coords


def benchmark_function(func, args, n_runs: int = 5, warmup: int = 1):
    """Benchmark a function with warmup runs."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)

        # Ensure GPU operations complete
        if GPU_AVAILABLE:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()

        times.append(time.perf_counter() - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
    }


def benchmark_standardize_matrix(n_genes: int, n_cells: int, n_runs: int = 5) -> Dict:
    """Benchmark standardize_matrix CPU vs GPU."""
    expr, _ = generate_test_data(n_genes, n_cells)

    # CPU benchmark
    cpu_stats = benchmark_function(
        lambda x: standardize_matrix(x, use_gpu=False),
        (expr,),
        n_runs=n_runs
    )

    results = {'cpu': cpu_stats, 'gpu': None, 'speedup': None}

    if GPU_AVAILABLE:
        import cupy as cp
        expr_gpu = cp.asarray(expr)

        gpu_stats = benchmark_function(
            lambda x: standardize_matrix(x, use_gpu=True),
            (expr_gpu,),
            n_runs=n_runs
        )
        results['gpu'] = gpu_stats
        results['speedup'] = cpu_stats['mean'] / gpu_stats['mean']

    return results


def benchmark_gaussian_weights(n_cells: int, radius: float = 100, n_runs: int = 5) -> Dict:
    """Benchmark Gaussian weight matrix creation CPU vs GPU."""
    _, coords = generate_test_data(100, n_cells)

    # CPU benchmark
    cpu_stats = benchmark_function(
        lambda c: create_gaussian_weights(c, radius=radius, use_gpu=False),
        (coords,),
        n_runs=n_runs
    )

    results = {'cpu': cpu_stats, 'gpu': None, 'speedup': None}

    if GPU_AVAILABLE:
        import cupy as cp
        coords_gpu = cp.asarray(coords)

        gpu_stats = benchmark_function(
            lambda c: create_gaussian_weights(c, radius=radius, use_gpu=True),
            (coords_gpu,),
            n_runs=n_runs
        )
        results['gpu'] = gpu_stats
        results['speedup'] = cpu_stats['mean'] / gpu_stats['mean']

    return results


def benchmark_spatial_lag(n_cells: int, radius: float = 100, n_runs: int = 5) -> Dict:
    """Benchmark spatial lag computation CPU vs GPU."""
    expr, coords = generate_test_data(1, n_cells)
    z = expr[0]

    # Create weight matrix (CPU)
    W = create_gaussian_weights(coords, radius=radius, use_gpu=False)

    # CPU benchmark
    cpu_stats = benchmark_function(
        lambda w, x: compute_spatial_lag(w, x, use_gpu=False),
        (W, z),
        n_runs=n_runs
    )

    results = {'cpu': cpu_stats, 'gpu': None, 'speedup': None}

    if GPU_AVAILABLE:
        import cupy as cp
        from cupyx.scipy import sparse as cp_sparse

        z_gpu = cp.asarray(z)
        W_gpu = cp_sparse.csr_matrix(W)

        gpu_stats = benchmark_function(
            lambda w, x: compute_spatial_lag(w, x, use_gpu=True),
            (W_gpu, z_gpu),
            n_runs=n_runs
        )
        results['gpu'] = gpu_stats
        results['speedup'] = cpu_stats['mean'] / gpu_stats['mean']

    return results


def benchmark_pairwise_moran(n_genes: int, n_cells: int, radius: float = 100, n_runs: int = 3) -> Dict:
    """Benchmark pairwise Moran's I computation CPU vs GPU."""
    expr, coords = generate_test_data(n_genes, n_cells)

    # CPU benchmark
    cpu_stats = benchmark_function(
        lambda: pairwise_moran(expr, coords, radius=radius, use_gpu=False),
        (),
        n_runs=n_runs
    )

    results = {'cpu': cpu_stats, 'gpu': None, 'speedup': None}

    if GPU_AVAILABLE:
        gpu_stats = benchmark_function(
            lambda: pairwise_moran(expr, coords, radius=radius, use_gpu=True),
            (),
            n_runs=n_runs
        )
        results['gpu'] = gpu_stats
        results['speedup'] = cpu_stats['mean'] / gpu_stats['mean']

    return results


def run_all_benchmarks(
    n_genes_list: List[int] = [100, 500, 1000],
    n_cells_list: List[int] = [1000, 5000, 10000],
    radius: float = 100,
    n_runs: int = 3,
) -> Dict:
    """Run all benchmarks with various data sizes."""
    results = {
        'gpu_available': GPU_AVAILABLE,
        'benchmarks': {}
    }

    print(f"GPU Available: {GPU_AVAILABLE}")
    print("=" * 60)

    # Standardize matrix benchmarks
    print("\n[1/4] Benchmarking standardize_matrix...")
    results['benchmarks']['standardize_matrix'] = {}
    for n_genes in n_genes_list:
        for n_cells in n_cells_list:
            key = f"{n_genes}x{n_cells}"
            print(f"  {key}...", end=" ", flush=True)
            result = benchmark_standardize_matrix(n_genes, n_cells, n_runs)
            results['benchmarks']['standardize_matrix'][key] = result

            if result['speedup']:
                print(f"Speedup: {result['speedup']:.1f}x")
            else:
                print(f"CPU: {result['cpu']['mean']*1000:.1f}ms")

    # Gaussian weights benchmarks
    print("\n[2/4] Benchmarking gaussian_weights...")
    results['benchmarks']['gaussian_weights'] = {}
    for n_cells in n_cells_list:
        key = f"{n_cells}"
        print(f"  n_cells={key}...", end=" ", flush=True)
        result = benchmark_gaussian_weights(n_cells, radius, n_runs)
        results['benchmarks']['gaussian_weights'][key] = result

        if result['speedup']:
            print(f"Speedup: {result['speedup']:.1f}x")
        else:
            print(f"CPU: {result['cpu']['mean']*1000:.1f}ms")

    # Spatial lag benchmarks
    print("\n[3/4] Benchmarking spatial_lag...")
    results['benchmarks']['spatial_lag'] = {}
    for n_cells in n_cells_list:
        key = f"{n_cells}"
        print(f"  n_cells={key}...", end=" ", flush=True)
        result = benchmark_spatial_lag(n_cells, radius, n_runs)
        results['benchmarks']['spatial_lag'][key] = result

        if result['speedup']:
            print(f"Speedup: {result['speedup']:.1f}x")
        else:
            print(f"CPU: {result['cpu']['mean']*1000:.1f}ms")

    # Pairwise Moran benchmarks (smaller sizes due to O(n^2) complexity)
    print("\n[4/4] Benchmarking pairwise_moran...")
    results['benchmarks']['pairwise_moran'] = {}
    small_genes = [50, 100, 200]
    small_cells = [500, 1000, 2000]
    for n_genes in small_genes:
        for n_cells in small_cells:
            key = f"{n_genes}x{n_cells}"
            print(f"  {key}...", end=" ", flush=True)
            result = benchmark_pairwise_moran(n_genes, n_cells, radius, n_runs)
            results['benchmarks']['pairwise_moran'][key] = result

            if result['speedup']:
                print(f"Speedup: {result['speedup']:.1f}x")
            else:
                print(f"CPU: {result['cpu']['mean']*1000:.1f}ms")

    print("\n" + "=" * 60)
    print("Benchmarks complete!")

    return results


def print_summary(results: Dict):
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    if not results['gpu_available']:
        print("GPU not available. Showing CPU-only results.")

    for benchmark_name, benchmark_data in results['benchmarks'].items():
        print(f"\n{benchmark_name}:")
        print("-" * 40)
        print(f"{'Size':<15} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
        print("-" * 40)

        for size, data in benchmark_data.items():
            cpu_ms = data['cpu']['mean'] * 1000
            if data['gpu']:
                gpu_ms = data['gpu']['mean'] * 1000
                speedup = data['speedup']
                print(f"{size:<15} {cpu_ms:<12.2f} {gpu_ms:<12.2f} {speedup:<10.1f}x")
            else:
                print(f"{size:<15} {cpu_ms:<12.2f} {'N/A':<12} {'N/A':<10}")


if __name__ == "__main__":
    results = run_all_benchmarks(
        n_genes_list=[100, 500],
        n_cells_list=[1000, 5000],
        n_runs=3
    )
    print_summary(results)
