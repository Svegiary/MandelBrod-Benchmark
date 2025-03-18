# Multithreaded Mandelbrot Benchmark

A Python implementation of the Mandelbrot set calculation that utilizes multithreading to demonstrate parallel processing performance benefits.

## Description

This benchmark:

- Implements both single-threaded and multithreaded Mandelbrot set calculations
- Measures performance with different numbers of threads
- Generates visualizations of the Mandelbrot set for different regions
- Creates performance comparison charts

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python mandelbrot_benchmark.py
```

The script will:

1. Run benchmarks using 1, 2, 4, 8, and N threads (where N is your CPU core count)
2. Generate Mandelbrot set visualizations for different regions
3. Save performance benchmark graphs

## Output Files

- `mandelbrot_full.png` - Full view of the Mandelbrot set
- `mandelbrot_zoom1.png` - Zoomed view of an interesting region
- `benchmark_full.png` - Performance comparison chart for the full view
- `benchmark_zoom1.png` - Performance comparison chart for the zoomed view
