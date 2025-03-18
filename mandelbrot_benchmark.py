#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import concurrent.futures
import multiprocessing

def mandelbrot(h, w, max_iters, y_min, y_max, x_min, x_max):
    """
    Calculate the Mandelbrot set for the given region and resolution.
    
    Args:
        h, w: Height and width of the resulting image
        max_iters: Maximum number of iterations for each point
        y_min, y_max, x_min, x_max: Boundaries of the region to calculate
    
    Returns:
        2D numpy array of iteration counts
    """
    y, x = np.ogrid[y_min:y_max:h*1j, x_min:x_max:w*1j]
    c = x + y*1j
    z = c
    divtime = max_iters + np.zeros(z.shape, dtype=int)
    
    for i in range(max_iters):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2
        div_now = diverge & (divtime == max_iters)
        divtime[div_now] = i
        z[diverge] = 2
        
    return divtime

def mandelbrot_chunk(args):
    """Process a horizontal chunk of the Mandelbrot image"""
    y_start, y_end, h_chunk, w, max_iters, y_min, y_max, x_min, x_max = args
    
    # Calculate the y range for this chunk
    y_chunk_min = y_min + (y_max - y_min) * y_start / h_chunk
    y_chunk_max = y_min + (y_max - y_min) * y_end / h_chunk
    
    return mandelbrot(y_end - y_start, w, max_iters, y_chunk_min, y_chunk_max, x_min, x_max)

def mandelbrot_parallel(h, w, max_iters, y_min, y_max, x_min, x_max, num_threads=None):
    """Calculate the Mandelbrot set using multiple threads"""
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()
    
    # Split the image into horizontal chunks
    chunk_size = h // num_threads
    chunks = []
    
    for i in range(num_threads):
        y_start = i * chunk_size
        y_end = (i + 1) * chunk_size if i < num_threads - 1 else h
        chunks.append((y_start, y_end, h, w, max_iters, y_min, y_max, x_min, x_max))
    
    # Process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(mandelbrot_chunk, chunks))
    
    # Combine the results
    combined = np.vstack(results)
    return combined

def benchmark(h, w, max_iters, y_min, y_max, x_min, x_max):
    """Run benchmarks with different numbers of threads"""
    
    # First run a single-threaded version
    print("Running single-threaded benchmark...")
    start = time.time()
    mandelbrot(h, w, max_iters, y_min, y_max, x_min, x_max)
    single_thread_time = time.time() - start
    print(f"Single-threaded time: {single_thread_time:.3f} seconds")
    
    thread_counts = [2, 4, 8, max(10, multiprocessing.cpu_count())]
    times = []
    
    for num_threads in thread_counts:
        print(f"Running with {num_threads} threads...")
        start = time.time()
        mandelbrot_parallel(h, w, max_iters, y_min, y_max, x_min, x_max, num_threads)
        thread_time = time.time() - start
        times.append(thread_time)
        print(f"{num_threads} threads: {thread_time:.3f} seconds, speedup: {single_thread_time/thread_time:.2f}x")
    
    return single_thread_time, thread_counts, times

def visualize_mandelbrot(image, h, w, y_min, y_max, x_min, x_max, filename=None):
    """Create a visualization of the Mandelbrot set"""
    # Create a custom colormap - black for points in the set, colored gradient for outside
    colors = [(0, 0, 0), (0, 0, 0.5), (0, 0.5, 1), (1, 1, 1), (1, 0.5, 0), (0.5, 0, 0), (0, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('mandelbrot', colors, N=256)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(image, cmap=cmap, origin='lower', extent=(x_min, x_max, y_min, y_max))
    plt.title(f'Mandelbrot Set ({h}x{w}, region: [{x_min}, {x_max}] x [{y_min}, {y_max}])')
    plt.colorbar(label='Iterations')
    
    if filename:
        plt.savefig(filename, dpi=150)
    else:
        plt.show()

def main():
    # Parameters - increased resolution and iterations for a more demanding benchmark
    h, w = 4000, 5000  # Image height and width (doubled from original)
    max_iters = 500     # Maximum iterations (5x more than original)
    
    # Interesting regions to explore - added more challenging regions
    regions = [
        # Full view
        {'y_min': -1.25, 'y_max': 1.25, 'x_min': -2.0, 'x_max': 0.5, 'name': 'full'},
        # Zoomed in area 
        {'y_min': -0.15, 'y_max': 0.15, 'x_min': -0.8, 'x_max': -0.5, 'name': 'zoom1'},
        # Deep zoom to a complex area (Seahorse Valley) - computationally intensive
        {'y_min': -0.745, 'y_max': -0.735, 'x_min': 0.21, 'x_max': 0.22, 'name': 'seahorse'},
        # Spiral structure - complex area requiring many iterations
        {'y_min': -0.7472, 'y_max': -0.7468, 'x_min': -0.1152, 'x_max': -0.1148, 'name': 'spiral'}
    ]
    
    all_results = []
    
    for region in regions:
        print(f"\nBenchmarking region: {region['name']}")
        y_min, y_max = region['y_min'], region['y_max']
        x_min, x_max = region['x_min'], region['x_max']
        
        # Run benchmark
        single_time, thread_counts, times = benchmark(h, w, max_iters, y_min, y_max, x_min, x_max)
        
        # Generate the final image using all available cores
        cpu_count = multiprocessing.cpu_count()
        print(f"\nGenerating final image using {cpu_count} threads...")
        start = time.time()
        image = mandelbrot_parallel(h, w, max_iters, y_min, y_max, x_min, x_max, cpu_count)
        print(f"Image generation time: {time.time() - start:.3f} seconds")
        
        # Save the image
        filename = f"mandelbrot_{region['name']}.png"
        visualize_mandelbrot(image, h, w, y_min, y_max, x_min, x_max, filename)
        print(f"Image saved as {filename}")
        
        # Plot benchmark results
        plt.figure(figsize=(10, 6))
        plt.plot([1] + thread_counts, [single_time] + times, 'o-')
        plt.axhline(y=single_time, color='r', linestyle='--', alpha=0.3)
        plt.title(f'Mandelbrot Benchmark - Region: {region["name"]}')
        plt.xlabel('Number of threads')
        plt.ylabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        for i, (threads, time_val) in enumerate(zip([1] + thread_counts, [single_time] + times)):
            plt.annotate(f'{time_val:.2f}s', (threads, time_val), 
                         textcoords="offset points", xytext=(0,10), ha='center')
        plt.savefig(f"benchmark_{region['name']}.png")
        
        # Store results for summary
        best_time = min(times)
        best_threads = thread_counts[times.index(best_time)]
        speedup = single_time / best_time
        all_results.append({
            'region': region['name'],
            'single_thread': single_time,
            'best_time': best_time,
            'best_threads': best_threads,
            'speedup': speedup
        })
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Region':<10} {'Single (s)':<12} {'Best (s)':<12} {'Threads':<10} {'Speedup':<10}")
    print("-"*60)
    
    total_speedup = 0
    for result in all_results:
        print(f"{result['region']:<10} {result['single_thread']:<12.3f} {result['best_time']:<12.3f} {result['best_threads']:<10} {result['speedup']:<10.2f}x")
        total_speedup += result['speedup']
    
    avg_speedup = total_speedup / len(all_results)
    print("-"*60)
    print(f"Average speedup across all regions: {avg_speedup:.2f}x")
    print(f"CPU count: {multiprocessing.cpu_count()}")
    print(f"Total computation: {h}x{w} pixels × {max_iters} max iterations × {len(regions)} regions")
    print("="*60)

if __name__ == "__main__":
    main() 