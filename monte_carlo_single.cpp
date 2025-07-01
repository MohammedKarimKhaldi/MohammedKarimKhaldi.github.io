#include <iostream>
#include <random>
#include <cmath>
#include <chrono>

int main(int argc, char* argv[]) {
    double S0 = 100, K = 100, r = 0.05, sigma = 0.2, T = 1.0;
    int N = 10000000;
    if (argc > 1) N = std::stoi(argv[1]);
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);
    double payoff_sum = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        double Z = dist(gen);
        double ST = S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
        double payoff = std::max(ST - K, 0.0);
        payoff_sum += payoff;
    }
    double price = std::exp(-r * T) * (payoff_sum / N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "C++ (single): Price = " << price << ", Time = " << elapsed.count() << "s\n";
    return 0;
} 