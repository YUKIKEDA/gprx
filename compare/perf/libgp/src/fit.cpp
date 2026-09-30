// B1-1 real-dataset cell: ARD SE + noise, Rprop as libgp ships it, then score.
// One fit per process. `native` runs Rprop for 100 iterations (its default);
// `matched` is N/A (libgp has no swappable optimizer); `fixed` skips it.
#include "gp.h"
#include "rprop.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

using json = nlohmann::json;

namespace {

constexpr double kZ95 = 1.959963984540054;
constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kRpropIterations = 100;

uint64_t peak_rss_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) == 0) {
        throw std::runtime_error("GetProcessMemoryInfo failed");
    }
    return static_cast<uint64_t>(counters.PeakWorkingSetSize);
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage failed");
    }
#ifdef __APPLE__
    return static_cast<uint64_t>(usage.ru_maxrss);
#else
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
#endif
}

Eigen::MatrixXd unpack_rows(const std::vector<double>& packed, int n_rows, int n_cols) {
    Eigen::MatrixXd x(n_rows, n_cols);
    for (int col = 0; col < n_cols; ++col) {
        for (int row = 0; row < n_rows; ++row) {
            x(row, col) = packed[static_cast<size_t>(col * n_rows + row)];
        }
    }
    return x;
}

json na_row(const json& c, const std::string& note) {
    return {{"lib", "libgp"},         {"name", c.at("name")},   {"status", "na"},
            {"protocol", c.at("protocol")}, {"fit_s", nullptr}, {"predict_s", nullptr},
            {"joint_evals", nullptr}, {"value_evals", nullptr}, {"iterations", nullptr},
            {"nlml", nullptr},        {"rmse", nullptr},        {"nlpd", nullptr},
            {"coverage95", nullptr},  {"peak_rss_bytes", nullptr}, {"note", note}};
}

struct Fitted {
    std::unique_ptr<libgp::GaussianProcess> gp;
    uint64_t evals = 0;
    double fit_s = 0.0;
};

// One timed fit: build, add the points, and (for `native`) run Rprop.
Fitted fit_gp(const json& c, const std::string& protocol, const Eigen::MatrixXd& x,
              const Eigen::VectorXd& y, int n, int d) {
    Fitted out;
    out.gp = std::make_unique<libgp::GaussianProcess>(static_cast<size_t>(d),
                                                     "CovSum ( CovSEard, CovNoise)");
    libgp::GaussianProcess& gp = *out.gp;
    Eigen::VectorXd loghyper(static_cast<Eigen::Index>(gp.covf().get_param_dim()));
    if (loghyper.size() != d + 2) {
        throw std::runtime_error("unexpected CovSEard+Noise parameter dimension");
    }
    // libgp keeps log ell, log sf (an amplitude), log sn (a noise std).
    for (int j = 0; j < d; ++j) {
        loghyper(j) = std::log(c.at("lengthscale_init").get<double>());
    }
    loghyper(d) = 0.5 * std::log(c.at("signal_variance_init").get<double>());
    loghyper(d + 1) = 0.5 * std::log(c.at("noise_variance_init").get<double>());
    gp.covf().set_loghyper(loghyper);

    const auto start = std::chrono::steady_clock::now();
    // Not `add_patterns(x, y)`: it hands libgp `x.row(i).data()`, a strided
    // pointer into the column-major matrix, which is only a point when d = 1.
    std::vector<double> point(static_cast<size_t>(d));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            point[static_cast<size_t>(j)] = x(i, j);
        }
        gp.add_pattern(point.data(), y(i));
    }
    if (protocol == "native") {
        libgp::RProp rprop;
        rprop.init();
        rprop.maximize(&gp, kRpropIterations, false);
        out.evals = kRpropIterations;  // eps_stop = 0: one gradient and one value per iteration
    } else {
        (void)gp.log_likelihood();
    }
    out.fit_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return out;
}

std::size_t warmup_fits(int n_rows) {
    const char* raw = std::getenv("PERF_WARMUP");
    if (raw != nullptr && raw[0] != '\0') {
        return static_cast<std::size_t>(std::strtoul(raw, nullptr, 10));
    }
    return n_rows <= 5000 ? 1 : 0;
}

json run(const json& c) {
    const std::string protocol = c.at("protocol").get<std::string>();
    if (protocol == "matched") {
        return na_row(c, "libgp has no swappable optimizer (Rprop only)");
    }
    const int n = c.at("n_rows").get<int>();
    const int d = c.at("n_cols").get<int>();
    const int m = c.at("xs_n_rows").get<int>();
    const auto x = unpack_rows(c.at("x").get<std::vector<double>>(), n, d);
    const auto xs = unpack_rows(c.at("xs").get<std::vector<double>>(), m, d);
    const auto y_raw = c.at("y").get<std::vector<double>>();
    const auto ys = c.at("ys").get<std::vector<double>>();
    const double y_mean = c.at("y_mean").get<double>();
    const double y_std = c.at("y_std").get<double>();
    Eigen::VectorXd y =
        Eigen::Map<const Eigen::VectorXd>(y_raw.data(), static_cast<Eigen::Index>(y_raw.size()));

    std::cerr << "PHASE fit" << std::endl;
    if (protocol != "native" && protocol != "fixed") {
        return na_row(c, "unknown protocol " + protocol);
    }
    for (std::size_t i = 0; i < warmup_fits(n); ++i) {
        std::cerr << "PHASE warmup" << std::endl;
        (void)fit_gp(c, protocol, x, y, n, d);
    }
    std::cerr << "PHASE fit" << std::endl;
    Fitted fitted = fit_gp(c, protocol, x, y, n, d);
    libgp::GaussianProcess& gp = *fitted.gp;
    const double fit_s = fitted.fit_s;
    const uint64_t evals = fitted.evals;
    const double nlml = -gp.log_likelihood();

    std::cerr << "PHASE predict" << std::endl;
    const auto pstart = std::chrono::steady_clock::now();
    const Eigen::MatrixXd pred = gp.predict(xs, true);
    const double predict_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - pstart).count();

    // `predict` evaluates the covariance at two temporaries, so CovNoise (which
    // adds only for the same object) contributes nothing: its variance is the
    // latent one. Add the fitted noise back to score the observation variance.
    const double noise_variance = std::exp(2.0 * gp.covf().get_loghyper()(d + 1));
    double sq = 0.0, nlpd = 0.0, inside = 0.0;
    for (int i = 0; i < m; ++i) {
        const double mu = pred(i, 0) * y_std + y_mean;
        const double var = (pred(i, 1) + noise_variance) * y_std * y_std;
        const double err = ys[static_cast<size_t>(i)] - mu;
        sq += err * err;
        nlpd += 0.5 * std::log(2.0 * kPi * var) + 0.5 * err * err / var;
        if (std::abs(err) <= kZ95 * std::sqrt(var)) {
            inside += 1.0;
        }
    }
    json out_row = {{"lib", "libgp"},
            {"name", c.at("name")},
            {"status", "ok"},
            {"protocol", protocol},
            {"fit_s", fit_s},
            {"predict_s", predict_s},
            {"joint_evals", evals},
            {"value_evals", evals},
            {"iterations", evals},
            {"nlml", nlml},
            {"rmse", std::sqrt(sq / m)},
            {"nlpd", nlpd / m},
            {"coverage95", inside / m},
            {"peak_rss_bytes", peak_rss_bytes()},
            {"note", protocol == "native" ? "Rprop 100 iterations" : "fixed"}};
    if (c.value("return_predictions", false)) {
        std::vector<double> mean, var;
        for (int i = 0; i < m; ++i) {
            mean.push_back(pred(i, 0) * y_std + y_mean);
            var.push_back((pred(i, 1) + noise_variance) * y_std * y_std);
        }
        out_row["pred_mean"] = mean;
        out_row["pred_var"] = var;
    }
    return out_row;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: libgp-fit CASE.json\n";
        return 2;
    }
    try {
        std::ifstream in(argv[1]);
        const json c = json::parse(in);
        std::cout << run(c).dump() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
