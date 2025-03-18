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
};

// Function to calculate the Mandelbrot set for a single point
int mandelbrot_point(std::complex<double> c, int max_iterations)
{
    std::complex<double> z(0, 0);
    int iterations = 0;

    while (std::abs(z) <= 2.0 && iterations < max_iterations)
    {
        z = z * z + c;
        iterations++;
    }

    return iterations;
}

// Function to calculate the Mandelbrot set for a range of points (single-threaded)
std::vector<std::vector<int>> mandelbrot(int height, int width, int max_iterations,
                                         double y_min, double y_max, double x_min, double x_max)
{
    std::vector<std::vector<int>> result(height, std::vector<int>(width));

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double real = x_min + (x_max - x_min) * x / width;
            double imag = y_min + (y_max - y_min) * y / height;
            std::complex<double> c(real, imag);
            result[y][x] = mandelbrot_point(c, max_iterations);
        }
    }

    return result;
}

// Function to calculate a chunk of the Mandelbrot set (for parallel processing)
void mandelbrot_chunk(int y_start, int y_end, int width, int max_iterations,
                      double y_min, double y_max, double x_min, double x_max,
                      std::vector<std::vector<int>> &result)
{
    for (int y = y_start; y < y_end; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double real = x_min + (x_max - x_min) * x / width;
            double imag = y_min + (y_max - y_min) * y / (result.size());
            std::complex<double> c(real, imag);
            result[y][x] = mandelbrot_point(c, max_iterations);
        }
    }
}

// Function to calculate the Mandelbrot set using multiple threads
std::vector<std::vector<int>> mandelbrot_parallel(int height, int width, int max_iterations,
                                                  double y_min, double y_max, double x_min, double x_max,
                                                  int num_threads)
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
                                      y_min, y_max, x_min, x_max, std::ref(result)));
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
            int value = data[y][x];

            // Create a color based on the number of iterations
            unsigned char r, g, b;
            if (value == max_iterations)
            {
                r = g = b = 0; // Black for points in the set
            }
            else
            {
                // Color gradient for points outside the set
                double normalized = static_cast<double>(value) / max_iterations;

                // Create a color gradient (similar to the Python version)
                if (normalized < 0.16)
                {
                    r = 0;
                    g = 0;
                    b = static_cast<unsigned char>(normalized / 0.16 * 128);
                }
                else if (normalized < 0.42)
                {
                    r = 0;
                    g = static_cast<unsigned char>((normalized - 0.16) / 0.26 * 128);
                    b = static_cast<unsigned char>(128 + (normalized - 0.16) / 0.26 * 127);
                }
                else if (normalized < 0.6425)
                {
                    r = static_cast<unsigned char>((normalized - 0.42) / 0.2225 * 255);
                    g = static_cast<unsigned char>(128 + (normalized - 0.42) / 0.2225 * 127);
                    b = 255;
                }
                else if (normalized < 0.8575)
                {
                    r = 255;
                    g = static_cast<unsigned char>(255 - (normalized - 0.6425) / 0.215 * 128);
                    b = static_cast<unsigned char>(255 - (normalized - 0.6425) / 0.215 * 255);
                }
                else
                {
                    r = static_cast<unsigned char>(255 - (normalized - 0.8575) / 0.1425 * 128);
                    g = static_cast<unsigned char>(127 - (normalized - 0.8575) / 0.1425 * 127);
                    b = 0;
                }
            }

            file.write(reinterpret_cast<char *>(&r), 1);
            file.write(reinterpret_cast<char *>(&g), 1);
            file.write(reinterpret_cast<char *>(&b), 1);
        }
    }

    file.close();
}

// Function to run benchmarks with different numbers of threads
BenchmarkResult benchmark(int height, int width, int max_iterations,
                          double y_min, double y_max, double x_min, double x_max,
                          const std::string &region_name)
{
    std::cout << "\nBenchmarking region: " << region_name << std::endl;

    // Run single-threaded version
    std::cout << "Running single-threaded benchmark..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    mandelbrot(height, width, max_iterations, y_min, y_max, x_min, x_max);
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
        mandelbrot_parallel(height, width, max_iterations, y_min, y_max, x_min, x_max, num_threads);
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

    return {region_name, single_thread_time, best_time, best_threads, speedup};
}

int main()
{
    // Parameters - high resolution for a challenging benchmark
    int height = 2000;
    int width = 2500;
    int max_iterations = 500;

    // Define interesting regions to explore
    std::vector<Region> regions = {
        // Full view
        {-1.25, 1.25, -2.0, 0.5, "full"},
        // Zoomed in area
        {-0.15, 0.15, -0.8, -0.5, "zoom1"},
        // Deep zoom to a complex area (Seahorse Valley)
        {-0.745, -0.735, 0.21, 0.22, "seahorse"},
        // Spiral structure
        {-0.7472, -0.7468, -0.1152, -0.1148, "spiral"}};

    std::vector<BenchmarkResult> all_results;

    for (const auto &region : regions)
    {
        // Run benchmark
        BenchmarkResult result = benchmark(height, width, max_iterations,
                                           region.y_min, region.y_max, region.x_min, region.x_max,
                                           region.name);
        all_results.push_back(result);

        // Generate the final image using all available cores
        int cpu_count = std::thread::hardware_concurrency();
        std::cout << "\nGenerating final image using " << cpu_count << " threads..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        auto image = mandelbrot_parallel(height, width, max_iterations,
                                         region.y_min, region.y_max, region.x_min, region.x_max,
                                         cpu_count);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Image generation time: "
                  << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;

        // Save the image
        std::string filename = "mandelbrot_" + region.name + ".ppm";
        save_image(image, max_iterations, filename);
        std::cout << "Image saved as " << filename << std::endl;
    }

    // Print summary
    std::cout << "\n"
              << std::string(60, '=') << std::endl;
    std::cout << "BENCHMARK SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << std::left << std::setw(10) << "Region"
              << std::setw(12) << "Single (s)"
              << std::setw(12) << "Best (s)"
              << std::setw(10) << "Threads"
              << std::setw(10) << "Speedup" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    double total_speedup = 0.0;
    for (const auto &result : all_results)
    {
        std::cout << std::left << std::setw(10) << result.region_name
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << result.single_thread_time
                  << std::setw(12) << result.best_time
                  << std::setw(10) << result.best_threads
                  << std::setw(10) << std::setprecision(2) << result.speedup << "x" << std::endl;
        total_speedup += result.speedup;
    }

    double avg_speedup = total_speedup / all_results.size();
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Average speedup across all regions: " << std::fixed << std::setprecision(2)
              << avg_speedup << "x" << std::endl;
    std::cout << "CPU count: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "Total computation: " << height << "x" << width << " pixels × "
              << max_iterations << " max iterations × " << regions.size() << " regions" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return 0;
}