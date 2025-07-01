#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>

double bootstrapSpot(int idx, const std::vector<double>& prices,
                     const std::vector<double>& coupons,
                     const std::vector<double>& spots,
                     double face) {
    double sum = 0.0;
    for (int j = 0; j < idx; ++j)
        sum += coupons[idx] / std::pow(1 + spots[j], j + 1);
    double numer = coupons[idx] + face;
    double denom = prices[idx] - sum;
    return std::pow(numer / denom, 1.0 / (idx + 1)) - 1.0;
}

int main() {
    std::vector<double> prices  = {98.0, 95.0, 92.0, 89.0, 86.0};
    std::vector<double> coupons = {2.0, 2.5, 3.0, 3.2, 3.5};
    double face = 100.0;
    std::vector<double> spots;

    std::ofstream out("spot_rates.csv");
    out << "Year,SpotRate\n";

    for (size_t i = 0; i < prices.size(); ++i) {
        double spot = bootstrapSpot(i, prices, coupons, spots, face);
        spots.push_back(spot);
        out << (i + 1) << "," << std::fixed << std::setprecision(6) << spot * 100 << "\n";
        std::cout << "Year " << (i + 1) << ": Spot rate = " << spot * 100 << "%\n";
    }

    out.close();
    return 0;
}
