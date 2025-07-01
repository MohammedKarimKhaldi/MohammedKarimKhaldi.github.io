#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <chrono>

int main(int argc, char* argv[]) {
    using namespace Eigen;
    double S0 = 100, K = 100, r = 0.05, sigma = 0.2, T = 1.0;
    int N = 10000000;
    if (argc > 1) N = std::stoi(argv[1]);

    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);

    ArrayXd Z(N);
    for (int i = 0; i < N; ++i) Z[i] = dist(gen);

    auto start = std::chrono::high_resolution_clock::now();
    ArrayXd ST = S0 * ((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z).exp();
    ArrayXd payoff = (ST - K).max(0.0);
    double price = std::exp(-r * T) * payoff.mean();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "C++ (Eigen vectorized): Price = " << price << ", Time = " << elapsed.count() << "s\n";
    return 0;
} 