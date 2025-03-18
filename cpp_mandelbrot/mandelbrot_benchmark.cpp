#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <thread>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <algorithm>
#include <functional>
#include <cmath>

// Define a structure for benchmark results
struct BenchmarkResult
{
    std::string region_name;
    double single_thread_time;
    double best_time;
    int best_threads;
    double speedup;
};

// Define a struct for region parameters
struct Region
{
    double y_min, y_max, x_min, x_max;
    std::string name;
    int extra_iterations; // Additional iterations for complex regions
};

// Define calculation methods
enum class CalcMethod
{
    STANDARD,     // Standard z = z^2 + c
    BURNING_SHIP, // Burning ship fractal
    MULTIBROT     // Higher power Multibrot set
};

// Function to calculate the Mandelbrot set for a single point with standard formula
int mandelbrot_point(std::complex<double> c, int max_iterations)
{
    std::complex<double> z(0, 0);
    int iterations = 0;

    while (std::abs(z) <= 2.0 && iterations < max_iterations)
    {
        z = z * z + c;
        iterations++;
    }

    // Smooth coloring algorithm (more CPU intensive)
    if (iterations < max_iterations)
    {
        // Add smoothing for more precise color transitions
        double log_zn = log(std::abs(z));
        double nu = log(log_zn / log(2.0)) / log(2.0);
        iterations = iterations + 1 - nu;
    }

    return iterations;
}

// Function to calculate the Burning Ship fractal (more compute-intensive variant)
int burning_ship_point(std::complex<double> c, int max_iterations)
{
    double real = 0;
    double imag = 0;
    int iterations = 0;

    while (real * real + imag * imag <= 4.0 && iterations < max_iterations)
    {
        double next_real = real * real - imag * imag + c.real();
        double next_imag = 2.0 * std::abs(real * imag) + c.imag();
        real = next_real;
        imag = next_imag;
        iterations++;
    }

    // Smooth coloring
    if (iterations < max_iterations)
    {
        double zn = sqrt(real * real + imag * imag);
        double nu = log(log(zn)) / log(2.0);
        iterations = iterations + 1 - nu;
    }

    return iterations;
}

// Function to calculate the Multibrot set (higher power z^n + c, more CPU intensive)
int multibrot_point(std::complex<double> c, int max_iterations, int power = 4)
{
    std::complex<double> z(0, 0);
    int iterations = 0;

    while (std::abs(z) <= 2.0 && iterations < max_iterations)
    {
        // Compute z^power (more expensive than z^2)
        std::complex<double> z_powered = z;
        for (int i = 1; i < power; i++)
        {
            z_powered *= z;
        }
        z = z_powered + c;
        iterations++;
    }

    // Smooth coloring
    if (iterations < max_iterations)
    {
        double log_zn = log(std::abs(z));
        double nu = log(log_zn / log(2.0)) / log(2.0);
        iterations = iterations + 1 - nu;
    }

    return iterations;
}

// Function to calculate a point using the selected method
int calculate_point(std::complex<double> c, int max_iterations, CalcMethod method, int power = 4)
{
    switch (method)
    {
    case CalcMethod::BURNING_SHIP:
        return burning_ship_point(c, max_iterations);
    case CalcMethod::MULTIBROT:
        return multibrot_point(c, max_iterations, power);
    case CalcMethod::STANDARD:
    default:
        return mandelbrot_point(c, max_iterations);
    }
}

// Function to calculate the Mandelbrot set for a range of points (single-threaded)
std::vector<std::vector<int>> mandelbrot(int height, int width, int max_iterations,
                                         double y_min, double y_max, double x_min, double x_max,
                                         CalcMethod method = CalcMethod::STANDARD)
{
    std::vector<std::vector<int>> result(height, std::vector<int>(width));

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double real = x_min + (x_max - x_min) * x / width;
            double imag = y_min + (y_max - y_min) * y / height;
            std::complex<double> c(real, imag);
            result[y][x] = calculate_point(c, max_iterations, method);
        }
    }

    return result;
}

// Function to calculate a chunk of the Mandelbrot set (for parallel processing)
void mandelbrot_chunk(int y_start, int y_end, int width, int max_iterations,
                      double y_min, double y_max, double x_min, double x_max,
                      std::vector<std::vector<int>> &result,
                      CalcMethod method = CalcMethod::STANDARD)
{
    for (int y = y_start; y < y_end; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double real = x_min + (x_max - x_min) * x / width;
            double imag = y_min + (y_max - y_min) * y / (result.size());
            std::complex<double> c(real, imag);
            result[y][x] = calculate_point(c, max_iterations, method);
        }
    }
}

// Function to calculate the Mandelbrot set using multiple threads
std::vector<std::vector<int>> mandelbrot_parallel(int height, int width, int max_iterations,
                                                  double y_min, double y_max, double x_min, double x_max,
                                                  int num_threads,
                                                  CalcMethod method = CalcMethod::STANDARD)
{
    std::vector<std::vector<int>> result(height, std::vector<int>(width));
    std::vector<std::thread> threads;

    // Calculate chunk size for each thread
    int chunk_size = height / num_threads;

    // Start threads
    for (int i = 0; i < num_threads; i++)
    {
        int y_start = i * chunk_size;
        int y_end = (i == num_threads - 1) ? height : (i + 1) * chunk_size;

        threads.push_back(std::thread(mandelbrot_chunk, y_start, y_end, width, max_iterations,
                                      y_min, y_max, x_min, x_max, std::ref(result), method));
    }

    // Join threads
    for (auto &t : threads)
    {
        t.join();
    }

    return result;
}

// Function to save the Mandelbrot set as a PPM image file
void save_image(const std::vector<std::vector<int>> &data, int max_iterations,
                const std::string &filename)
{
    int height = data.size();
    int width = data[0].size();

    std::ofstream file(filename, std::ios::binary);
    file << "P6\n"
         << width << " " << height << "\n255\n";

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double value = data[y][x];

            // More complex color calculation (more CPU intensive)
            unsigned char r, g, b;
            if (value >= max_iterations)
            {
                r = g = b = 0; // Black for points in the set
            }
            else
            {
                // Enhanced color gradient for points outside the set
                double normalized = value / max_iterations;

                // More complex HSV to RGB conversion (more computationally intensive)
                double h = 360.0 * normalized;
                double s = 1.0;
                double v = normalized < 0.5 ? normalized * 2 : 1.0;

                // HSV to RGB conversion
                double c = v * s;
                double x = c * (1 - std::abs(std::fmod(h / 60.0, 2) - 1));
                double m = v - c;

                double r1, g1, b1;
                if (h < 60)
                {
                    r1 = c;
                    g1 = x;
                    b1 = 0;
                }
                else if (h < 120)
                {
                    r1 = x;
                    g1 = c;
                    b1 = 0;
                }
                else if (h < 180)
                {
                    r1 = 0;
                    g1 = c;
                    b1 = x;
                }
                else if (h < 240)
                {
                    r1 = 0;
                    g1 = x;
                    b1 = c;
                }
                else if (h < 300)
                {
                    r1 = x;
                    g1 = 0;
                    b1 = c;
                }
                else
                {
                    r1 = c;
                    g1 = 0;
                    b1 = x;
                }

                r = static_cast<unsigned char>((r1 + m) * 255);
                g = static_cast<unsigned char>((g1 + m) * 255);
                b = static_cast<unsigned char>((b1 + m) * 255);
            }

            file.write(reinterpret_cast<char *>(&r), 1);
            file.write(reinterpret_cast<char *>(&g), 1);
            file.write(reinterpret_cast<char *>(&b), 1);
        }
    }

    file.close();
}

// String representation of calculation method
std::string method_to_string(CalcMethod method)
{
    switch (method)
    {
    case CalcMethod::BURNING_SHIP:
        return "Burning Ship";
    case CalcMethod::MULTIBROT:
        return "Multibrot";
    case CalcMethod::STANDARD:
        return "Standard";
    default:
        return "Unknown";
    }
}

// Function to run benchmarks with different numbers of threads
BenchmarkResult benchmark(int height, int width, int max_iterations,
                          double y_min, double y_max, double x_min, double x_max,
                          const std::string &region_name,
                          CalcMethod method = CalcMethod::STANDARD)
{
    std::cout << "\nBenchmarking region: " << region_name << " (" << method_to_string(method) << ")" << std::endl;

    // Run single-threaded version
    std::cout << "Running single-threaded benchmark..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    mandelbrot(height, width, max_iterations, y_min, y_max, x_min, x_max, method);
    auto end = std::chrono::high_resolution_clock::now();
    double single_thread_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Single-threaded time: " << single_thread_time << " seconds" << std::endl;

    // Run multithreaded versions with different thread counts
    std::vector<int> thread_counts = {2, 4, 8, std::max(10, static_cast<int>(std::thread::hardware_concurrency()))};
    std::vector<double> times;

    for (int num_threads : thread_counts)
    {
        std::cout << "Running with " << num_threads << " threads..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        mandelbrot_parallel(height, width, max_iterations, y_min, y_max, x_min, x_max, num_threads, method);
        end = std::chrono::high_resolution_clock::now();
        double thread_time = std::chrono::duration<double>(end - start).count();
        times.push_back(thread_time);
        std::cout << num_threads << " threads: " << thread_time << " seconds, speedup: "
                  << single_thread_time / thread_time << "x" << std::endl;
    }

    // Find the best performance
    auto min_it = std::min_element(times.begin(), times.end());
    double best_time = *min_it;
    int best_threads = thread_counts[std::distance(times.begin(), min_it)];
    double speedup = single_thread_time / best_time;

    return {region_name + " (" + method_to_string(method) + ")", single_thread_time, best_time, best_threads, speedup};
}

int main()
{
    // Parameters - increased resolution and iterations for an even more challenging benchmark
    int height = 4000;         // 2x the original resolution
    int width = 5000;          // 2x the original resolution
    int max_iterations = 2000; // 4x the original iteration count

    // Define interesting regions to explore with extra iterations for complex areas
    std::vector<Region> regions = {
        // Full view
        {-1.25, 1.25, -2.0, 0.5, "full", 0},
        // Deep zoom to a complex area (Seahorse Valley)
        {-0.745, -0.735, 0.21, 0.22, "seahorse", 500},
        // Feigenbaum point - very computationally intensive
        {-1.401155, -1.401150, 0.0000, 0.0000045, "feigenbaum", 1000},
        // Mini Mandelbrot deep zoom - extremely computationally intensive
        {-1.77777515, -1.77777505, 0.0117, 0.0118, "mini", 2000},
        // Spiral structure with very fine detail
        {-0.747203, -0.747196, -0.1151, -0.1150, "spiral", 1500}};

    // Define calculation methods to benchmark (increasing complexity)
    std::vector<CalcMethod> methods = {
        CalcMethod::STANDARD,     // Original Mandelbrot
        CalcMethod::BURNING_SHIP, // Burning Ship (more CPU intensive)
        CalcMethod::MULTIBROT     // Multibrot with higher powers (most CPU intensive)
    };

    std::vector<BenchmarkResult> all_results;

    for (const auto &method : methods)
    {
        for (const auto &region : regions)
        {
            // Adjust max iterations based on region complexity
            int iterations = max_iterations + region.extra_iterations;

            // Run benchmark (skip some combinations to save time)
            if (method == CalcMethod::MULTIBROT &&
                (region.name == "feigenbaum" || region.name == "mini"))
            {
                std::cout << "Skipping extremely intensive " << method_to_string(method)
                          << " calculation for " << region.name << std::endl;
                continue;
            }

            BenchmarkResult result = benchmark(height, width, iterations,
                                               region.y_min, region.y_max, region.x_min, region.x_max,
                                               region.name, method);
            all_results.push_back(result);

            // Generate the final image using all available cores
            if (method == CalcMethod::STANDARD)
            { // Only save images for standard method
                int cpu_count = std::thread::hardware_concurrency();
                std::cout << "\nGenerating final image using " << cpu_count << " threads..." << std::endl;
                auto start = std::chrono::high_resolution_clock::now();
                auto image = mandelbrot_parallel(height, width, iterations,
                                                 region.y_min, region.y_max, region.x_min, region.x_max,
                                                 cpu_count, method);
                auto end = std::chrono::high_resolution_clock::now();
                std::cout << "Image generation time: "
                          << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;

                // Save the image
                std::string filename = "mandelbrot_" + region.name + ".ppm";
                save_image(image, iterations, filename);
                std::cout << "Image saved as " << filename << std::endl;
            }
        }
    }

    // Print summary
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "BENCHMARK SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::left << std::setw(25) << "Region"
              << std::setw(12) << "Single (s)"
              << std::setw(12) << "Best (s)"
              << std::setw(10) << "Threads"
              << std::setw(10) << "Speedup" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    double total_speedup = 0.0;
    for (const auto &result : all_results)
    {
        std::cout << std::left << std::setw(25) << result.region_name
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << result.single_thread_time
                  << std::setw(12) << result.best_time
                  << std::setw(10) << result.best_threads
                  << std::setw(10) << std::setprecision(2) << result.speedup << "x" << std::endl;
        total_speedup += result.speedup;
    }

    double avg_speedup = total_speedup / all_results.size();
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Average speedup across all regions: " << std::fixed << std::setprecision(2)
              << avg_speedup << "x" << std::endl;
    std::cout << "CPU count: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "Total computation: " << height << "x" << width << " pixels × "
              << "variable max iterations × " << regions.size() << " regions × "
              << methods.size() << " methods" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    return 0;
}