import numpy as np
import time
import subprocess
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import re
import os

S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
iteration_counts = [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000]

def mc_worker(args):
    N, S0, K, r, sigma, T = args
    Z = np.random.normal(0, 1, N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    return np.sum(payoff)

def parse_cpp_time(output):
    output = output.strip()
    # Accepts both normal and scientific notation
    match = re.search(r'Time = ([0-9.eE+-]+)s', output)
    if match:
        try:
            return float(match.group(1))
        except Exception as e:
            print(f"[Warning] Could not convert time to float: {match.group(1)} ({e})")
            return None
    else:
        print(f"[Warning] Could not parse time from output: {output}")
        return None

def main():
    # Single-threaded Python
    python_times = []
    for N in iteration_counts:
        start = time.time()
        Z = np.random.normal(0, 1, N)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        payoff = np.maximum(ST - K, 0)
        price = np.exp(-r * T) * np.mean(payoff)
        elapsed = time.time() - start
        python_times.append(elapsed)
        print(f"Python single N={N}: {elapsed:.3f}s, price={price:.2f}")

    # Parallel Python (multiprocessing)
    parallel_times = []
    num_workers = min(cpu_count(), 8)
    for N in iteration_counts:
        chunk = N // num_workers
        args = [(chunk, S0, K, r, sigma, T) for _ in range(num_workers)]
        start = time.time()
        with Pool(num_workers) as pool:
            results = pool.map(mc_worker, args)
        payoff_sum = sum(results)
        price = np.exp(-r * T) * (payoff_sum / N)
        elapsed = time.time() - start
        parallel_times.append(elapsed)
        print(f"Python parallel N={N}: {elapsed:.3f}s, price={price:.2f}")

    # Single-threaded C++
    cpp_times = []
    for N in iteration_counts:
        cmd = f'./monte_carlo_single {N}'
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        cpp_time = parse_cpp_time(result.stdout)
        cpp_times.append(cpp_time)
        print(f"C++ single N={N}: {cpp_time if cpp_time is not None else 'N/A'}s, output: {result.stdout.strip()}")

    # Eigen vectorized C++
    eigen_times = []
    for N in iteration_counts:
        cmd = f'./monte_carlo_eigen {N}'
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        eigen_time = parse_cpp_time(result.stdout)
        eigen_times.append(eigen_time)
        print(f"C++ Eigen N={N}: {eigen_time if eigen_time is not None else 'N/A'}s, output: {result.stdout.strip()}")

    # OpenMP C++ (skip if executable is missing)
    openmp_times = []
    if os.path.isfile('./monte_carlo_openmp'):
        for N in iteration_counts:
            cmd = f'./monte_carlo_openmp {N}'
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            openmp_time = parse_cpp_time(result.stdout)
            openmp_times.append(openmp_time)
            print(f"C++ OpenMP N={N}: {openmp_time if openmp_time is not None else 'N/A'}s, output: {result.stdout.strip()}")
    else:
        print("[Info] Skipping OpenMP C++: Executable not found.")
        openmp_times = [None] * len(iteration_counts)

    # Plot (skip None values)
    def safe_plot(x, y, *args, **kwargs):
        x_plot = [xi for xi, yi in zip(x, y) if yi is not None]
        y_plot = [yi for yi in y if yi is not None]
        plt.plot(x_plot, y_plot, *args, **kwargs)

    plt.figure(figsize=(8,6))
    safe_plot(iteration_counts, python_times, 'o-', label='Python (single-threaded)')
    safe_plot(iteration_counts, parallel_times, 'o-', label='Python (parallel)')
    safe_plot(iteration_counts, cpp_times, 'o-', label='C++ (single-threaded)')
    safe_plot(iteration_counts, eigen_times, 'o-', label='C++ (Eigen vectorized)')
    safe_plot(iteration_counts, openmp_times, 'o-', label='C++ (OpenMP)')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Time (seconds)')
    plt.title('Monte Carlo Option Pricing: Runtime Comparison')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('projects/cpp-implementations/monte_carlo_runtime_comparison.png')
    plt.show()

if __name__ == "__main__":
    main() 