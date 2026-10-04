#include "gp.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

std::size_t env_size(const char* key, std::size_t fallback) {
    const char* raw = std::getenv(key);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (end == raw) {
        return fallback;
    }
    return static_cast<std::size_t>(parsed);
}

std::size_t warmup_count() {
    return env_size("PERF_WARMUP", 1);
}

std::size_t default_reps(int n_rows) {
    if (n_rows <= 256) {
        return 51;
    }
    if (n_rows <= 1024) {
        return 21;
    }
    return 7;
}

std::size_t timed_reps(int n_rows) {
    const char* raw = std::getenv("PERF_REPS");
    if (raw != nullptr && raw[0] != '\0') {
        const std::size_t reps = env_size("PERF_REPS", 7);
        return reps == 0 ? 1 : reps;
    }
    return default_reps(n_rows);
}

double median(std::vector<double> samples) {
    if (samples.empty()) {
        throw std::runtime_error("median of empty samples");
    }
    std::sort(samples.begin(), samples.end());
    const std::size_t n = samples.size();
    if (n % 2 == 1) {
        return samples[n / 2];
    }
    return 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
}

std::pair<double, double> min_max(const std::vector<double>& samples) {
    if (samples.empty()) {
        throw std::runtime_error("min_max of empty samples");
    }
    const auto [lo, hi] = std::minmax_element(samples.begin(), samples.end());
    return {*lo, *hi};
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
#elif defined(__linux__)
    // Not ru_maxrss: exec folds the pre-exec (parent's) high-water mark into it.
    // VmHWM belongs to the current address space, which exec replaces.
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream fields(line.substr(6));
            uint64_t kib = 0;
            std::string unit;
            if (!(fields >> kib >> unit) || unit != "kB") {
                throw std::runtime_error("cannot parse VmHWM: " + line);
            }
            return kib * 1024ULL;
        }
    }
    throw std::runtime_error("VmHWM missing from /proc/self/status");
#elif defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage failed");
    }
    return static_cast<uint64_t>(usage.ru_maxrss);
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage failed");
    }
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
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
    const Eigen::MatrixXd x = unpack_rows(x_packed, n_rows, n_cols);
    const Eigen::VectorXd y = zscore(y_raw);
    const Eigen::MatrixXd xs = unpack_rows(xs_packed, xs_n_rows, xs_n_cols);
    const std::size_t warmup = warmup_count();
    const std::size_t reps = timed_reps(n_rows);

    std::vector<double> factor_samples;
    factor_samples.reserve(reps);
    std::unique_ptr<libgp::GaussianProcess> fitted;
    for (std::size_t i = 0; i < warmup + reps; ++i) {
        fitted.reset();
        auto next = std::make_unique<libgp::GaussianProcess>(static_cast<size_t>(n_cols), cov);
        next->covf().set_loghyper(loghyper);
        const auto start = std::chrono::steady_clock::now();
        next->add_patterns(x, y);
        const double dt =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (i >= warmup) {
            factor_samples.push_back(dt);
        }
        fitted = std::move(next);
    }

    std::vector<double> eval_samples;
    eval_samples.reserve(reps);
    for (std::size_t i = 0; i < warmup + reps; ++i) {
        // Same theta as after factor. Must run before the clock: libgp
        // skips `compute()` unless `loghyper_changed`, so this is the
        // cache-bust that matches gprx `value_and_gradient_into`.
        fitted->covf().set_loghyper(loghyper);
        const auto start = std::chrono::steady_clock::now();
        (void)fitted->log_likelihood_gradient();
        const double dt =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (i >= warmup) {
            eval_samples.push_back(dt);
        }
    }
    std::vector<double> eval_scale;
    eval_scale.reserve(eval_samples.size());
    for (double sample : eval_samples) {
        eval_scale.push_back(sample * static_cast<double>(n_evals));
    }

    std::vector<double> predict_samples;
    predict_samples.reserve(reps);
    for (std::size_t i = 0; i < warmup + reps; ++i) {
        const auto start = std::chrono::steady_clock::now();
        (void)fitted->predict(xs, true);
        const double dt =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (i >= warmup) {
            predict_samples.push_back(dt);
        }
    }

    const auto [factor_lo, factor_hi] = min_max(factor_samples);
    const auto [eval_lo, eval_hi] = min_max(eval_scale);
    const auto [predict_lo, predict_hi] = min_max(predict_samples);
    return {
        {"lib", "libgp"},
        {"name", name},
        {"status", "ok"},
        {"factor_s", median(factor_samples)},
        {"factor_min_s", factor_lo},
        {"factor_max_s", factor_hi},
        {"eval_s", median(eval_scale)},
        {"eval_min_s", eval_lo},
        {"eval_max_s", eval_hi},
        {"predict_s", median(predict_samples)},
        {"predict_min_s", predict_lo},
        {"predict_max_s", predict_hi},
        {"joint_evals", n_evals},
        {"peak_rss_bytes", peak_rss_bytes()},
        {"warmup", warmup},
        {"reps", reps},
        {"note", "add_patterns factor + N× median of one log_likelihood_gradient; discard+timed"},
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
