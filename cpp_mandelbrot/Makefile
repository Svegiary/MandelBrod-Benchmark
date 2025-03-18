CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -Wextra
LDFLAGS = -pthread

TARGET = mandelbrot_benchmark

all: $(TARGET)

$(TARGET): mandelbrot_benchmark.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET) *.ppm

.PHONY: all clean 