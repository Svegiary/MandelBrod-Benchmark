# Multithreaded Mandelbrot Benchmark (C++ Version)

A high-performance C++ implementation of the Mandelbrot set calculation that utilizes multithreading to demonstrate parallel processing benefits.

## Description

This benchmark:

- Implements both single-threaded and multithreaded Mandelbrot set calculations using C++ threads
- Measures performance with different numbers of threads
- Generates PPM images of the Mandelbrot set for different regions
- Provides a detailed performance summary

## Features

- Uses the C++ Standard Library for portability
- `std::thread` for multithreading
- `std::chrono` for precise timing measurements
- PPM image output (viewable in most image viewers)

## Requirements

- C++17 compatible compiler
- POSIX threads support (for Linux/macOS)
- For Windows: MinGW or Visual Studio with C++17 support

## Building

```bash
# Build the benchmark
make

# Clean the build
make clean
```

## Usage

```bash
# Run the benchmark
./mandelbrot_benchmark
```

The program will:

1. Run benchmarks using 1, 2, 4, 8, and N threads (where N is your CPU core count)
2. Generate PPM images of the Mandelbrot set for different regions
3. Display a performance summary of the results

## Output Files

- `mandelbrot_full.ppm` - Full view of the Mandelbrot set
- `mandelbrot_zoom1.ppm` - Zoomed view of an interesting region
- `mandelbrot_seahorse.ppm` - Seahorse Valley region
- `mandelbrot_spiral.ppm` - Spiral region

## Performance

The C++ implementation is significantly faster than the Python version due to:

1. Native code execution
2. More efficient memory management
3. Better compiler optimizations (-O3 and -march=native flags)

This benchmark is an excellent demonstration of multithreading benefits for CPU-intensive tasks.
