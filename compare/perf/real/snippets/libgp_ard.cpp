// The comparison regression, libgp: ARD SE, learned from a fixed start.
#include "gp.h"
#include "rprop.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    const int n = 200, m = 50, d = 3;
    std::mt19937 rng(0);
    std::normal_distribution<double> normal;
    Eigen::MatrixXd x(n, d), xs(m, d);
    for (int i = 0; i < n; ++i) for (int j = 0; j < d; ++j) x(i, j) = normal(rng);
    for (int i = 0; i < m; ++i) for (int j = 0; j < d; ++j) xs(i, j) = normal(rng);
    Eigen::VectorXd y(n), ys(m);
    for (int i = 0; i < n; ++i) y(i) = std::sin(x.row(i).sum()) + 0.1 * normal(rng);
    for (int i = 0; i < m; ++i) ys(i) = std::sin(xs.row(i).sum());

    // snippet:begin
    libgp::GaussianProcess gp(d, "CovSum ( CovSEard, CovNoise)");
    Eigen::VectorXd loghyper(d + 2);  // log ell (d of them), log sf, log sn
    loghyper << 0.0, 0.0, 0.0, 0.0, std::log(std::sqrt(0.1));
    gp.covf().set_loghyper(loghyper);
    for (int i = 0; i < n; ++i) {  // add_patterns(x, y) would read strided rows
        std::vector<double> row(d);
        for (int j = 0; j < d; ++j) row[j] = x(i, j);
        gp.add_pattern(row.data(), y(i));
    }
    libgp::RProp rprop;  // libgp's optimizer: resilient backpropagation
    rprop.init();
    rprop.maximize(&gp, 100, false);
    const Eigen::MatrixXd pred = gp.predict(xs, true);  // mean, latent variance
    const double noise = std::exp(2.0 * gp.covf().get_loghyper()(d + 1));
    double nlpd = 0.0;
    for (int i = 0; i < m; ++i) {
        const double v = pred(i, 1) + noise, e = ys(i) - pred(i, 0);
        nlpd += 0.5 * std::log(2.0 * M_PI * v) + 0.5 * e * e / v;
    }
    nlpd /= m;
    // snippet:end
    std::printf("libgp    mean[0]=%.4f std[0]=%.4f nlpd=%.4f\n", pred(0, 0), std::sqrt(pred(0, 1) + noise), nlpd);
    return 0;
}
