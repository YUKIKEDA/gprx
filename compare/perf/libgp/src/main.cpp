#include "gp.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
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

json na_row(const std::string& name, const std::string& note) {
    return {
        {"lib", "libgp"},
        {"name", name},
        {"status", "na"},
        {"factor_s", nullptr},
        {"eval_s", nullptr},
        {"predict_s", nullptr},
        {"joint_evals", nullptr},
        {"peak_rss_bytes", nullptr},
        {"note", note},
    };
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

Eigen::VectorXd zscore(const std::vector<double>& y) {
    Eigen::VectorXd v =
        Eigen::Map<const Eigen::VectorXd>(y.data(), static_cast<Eigen::Index>(y.size()));
    const double mean = v.mean();
    const double std =
        std::sqrt((v.array() - mean).square().mean());
    const double denom = std == 0.0 ? 1.0 : std;
    return (v.array() - mean) / denom;
}

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

json run(const json& case_json) {
    const std::string name = case_json.at("name").get<std::string>();
    const bool ard = case_json.at("ard").get<bool>();
    const int n_rows = case_json.at("n_rows").get<int>();
    const int n_cols = case_json.at("n_cols").get<int>();
    const int xs_n_rows = case_json.at("xs_n_rows").get<int>();
    const int xs_n_cols = case_json.at("xs_n_cols").get<int>();
    const auto x_packed = case_json.at("x").get<std::vector<double>>();
    const auto y_raw = case_json.at("y").get<std::vector<double>>();
    const auto xs_packed = case_json.at("xs").get<std::vector<double>>();
    const auto lengthscales = case_json.at("lengthscales_init").get<std::vector<double>>();
    const double noise_variance = case_json.at("noise_variance_init").get<double>();
    const uint64_t n_evals = case_json.at("joint_evals").get<uint64_t>();

    const std::string cov =
        ard ? "CovSum ( CovSEard, CovNoise)" : "CovSum ( CovSEiso, CovNoise)";
    libgp::GaussianProcess gp(static_cast<size_t>(n_cols), cov);
    Eigen::VectorXd loghyper(static_cast<Eigen::Index>(gp.covf().get_param_dim()));
    const double log_noise = std::log(std::sqrt(noise_variance));
    if (ard) {
        if (lengthscales.size() != static_cast<size_t>(n_cols)) {
            return na_row(name, "ARD lengthscale count does not match n_cols");
        }
        if (loghyper.size() != n_cols + 2) {
            return na_row(name, "unexpected CovSEard+Noise parameter dimension");
        }
        for (int d = 0; d < n_cols; ++d) {
            loghyper(d) = std::log(lengthscales[static_cast<size_t>(d)]);
        }
        loghyper(n_cols) = 0.0;
        loghyper(n_cols + 1) = log_noise;
    } else {
        if (lengthscales.empty()) {
            return na_row(name, "missing lengthscale");
        }
        if (loghyper.size() != 3) {
            return na_row(name, "unexpected CovSEiso+Noise parameter dimension");
        }
        loghyper << std::log(lengthscales[0]), 0.0, log_noise;
    }
    gp.covf().set_loghyper(loghyper);

    const Eigen::MatrixXd x = unpack_rows(x_packed, n_rows, n_cols);
    const Eigen::VectorXd y = zscore(y_raw);
    const Eigen::MatrixXd xs = unpack_rows(xs_packed, xs_n_rows, xs_n_cols);

    const auto factor_start = std::chrono::steady_clock::now();
    gp.add_patterns(x, y);
    const double factor_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - factor_start).count();

    const auto eval_start = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < n_evals; ++i) {
        gp.covf().set_loghyper(loghyper);
        (void)gp.log_likelihood_gradient();
    }
    const double eval_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - eval_start).count();

    const auto predict_start = std::chrono::steady_clock::now();
    (void)gp.predict(xs, true);
    const double predict_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - predict_start).count();

    return {
        {"lib", "libgp"},
        {"name", name},
        {"status", "ok"},
        {"factor_s", factor_s},
        {"eval_s", eval_s},
        {"predict_s", predict_s},
        {"joint_evals", n_evals},
        {"peak_rss_bytes", peak_rss_bytes()},
        {"note", "add_patterns factor + N log_likelihood_gradient at the same theta"},
    };
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: libgp-perf CASE.json\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "failed to open " << argv[1] << "\n";
        return 1;
    }
    json case_json;
    try {
        in >> case_json;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    json row;
    try {
        row = run(case_json);
    } catch (const std::exception& e) {
        const std::string name =
            case_json.contains("name") ? case_json["name"].get<std::string>() : std::string("unknown");
        row = na_row(name, e.what());
    }
    std::cout << row.dump() << "\n";
    return 0;
}
