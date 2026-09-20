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

Eigen::VectorXd loghyper_from_case(const json& case_json, libgp::GaussianProcess& gp) {
    const bool ard = case_json.at("ard").get<bool>();
    const int n_cols = case_json.at("n_cols").get<int>();
    const auto lengthscales = case_json.at("lengthscales_init").get<std::vector<double>>();
    const double noise_variance = case_json.at("noise_variance_init").get<double>();
    Eigen::VectorXd loghyper(static_cast<Eigen::Index>(gp.covf().get_param_dim()));
    const double log_noise = std::log(std::sqrt(noise_variance));
    if (ard) {
        if (lengthscales.size() != static_cast<size_t>(n_cols)) {
            throw std::runtime_error("ARD lengthscale count does not match n_cols");
        }
        if (loghyper.size() != n_cols + 2) {
            throw std::runtime_error("unexpected CovSEard+Noise parameter dimension");
        }
        for (int d = 0; d < n_cols; ++d) {
            loghyper(d) = std::log(lengthscales[static_cast<size_t>(d)]);
        }
        loghyper(n_cols) = 0.0;
        loghyper(n_cols + 1) = log_noise;
    } else {
        if (lengthscales.empty()) {
            throw std::runtime_error("missing lengthscale");
        }
        if (loghyper.size() != 3) {
            throw std::runtime_error("unexpected CovSEiso+Noise parameter dimension");
        }
        loghyper << std::log(lengthscales[0]), 0.0, log_noise;
    }
    return loghyper;
}

void add_point(
    libgp::GaussianProcess& gp,
    const std::vector<double>& x_packed,
    const std::vector<double>& y,
    int n_rows,
    int n_cols,
    int index
) {
    std::vector<double> x(static_cast<size_t>(n_cols));
    for (int d = 0; d < n_cols; ++d) {
        x[static_cast<size_t>(d)] =
            x_packed[static_cast<size_t>(d * n_rows + index)];
    }
    gp.add_pattern(x.data(), y[static_cast<size_t>(index)]);
}

json snapshot_query(
    libgp::GaussianProcess& gp,
    int n,
    double noise_variance,
    const std::vector<double>& xs_packed,
    int xs_n_rows,
    int xs_n_cols
) {
    json means = json::array();
    json vars = json::array();
    std::vector<double> x(static_cast<size_t>(xs_n_cols));
    for (int i = 0; i < xs_n_rows; ++i) {
        for (int d = 0; d < xs_n_cols; ++d) {
            x[static_cast<size_t>(d)] =
                xs_packed[static_cast<size_t>(d * xs_n_rows + i)];
        }
        means.push_back(gp.f(x.data()));
        vars.push_back(gp.var(x.data()) + noise_variance);
    }
    return {
        {"n", n},
        {"mean", means},
        {"observation_variance", vars},
        {"neg_log_marginal_likelihood", -gp.log_likelihood()},
    };
}

json run_golden(const json& case_json) {
    const std::string name = case_json.at("name").get<std::string>();
    const bool ard = case_json.at("ard").get<bool>();
    const int n_rows = case_json.at("n_rows").get<int>();
    const int n_cols = case_json.at("n_cols").get<int>();
    const int xs_n_rows = case_json.at("xs_n_rows").get<int>();
    const int xs_n_cols = case_json.at("xs_n_cols").get<int>();
    const auto x_packed = case_json.at("x").get<std::vector<double>>();
    const auto y = case_json.at("y").get<std::vector<double>>();
    const auto xs_packed = case_json.at("xs").get<std::vector<double>>();
    const double noise_variance = case_json.at("noise_variance_init").get<double>();
    const int start_n = case_json.value("start_n", 4);
    if (n_rows < start_n || start_n < 2) {
        throw std::runtime_error("start_n out of range");
    }

    const std::string cov =
        ard ? "CovSum ( CovSEard, CovNoise)" : "CovSum ( CovSEiso, CovNoise)";
    libgp::GaussianProcess gp(static_cast<size_t>(n_cols), cov);
    gp.covf().set_loghyper(loghyper_from_case(case_json, gp));

    json steps = json::array();
    for (int i = 0; i < n_rows; ++i) {
        add_point(gp, x_packed, y, n_rows, n_cols, i);
        if (i + 1 >= start_n) {
            json step = snapshot_query(
                gp, i + 1, noise_variance, xs_packed, xs_n_rows, xs_n_cols
            );
            steps.push_back(std::move(step));
        }
    }
    return {
        {"source", "libgp"},
        {"name", name},
        {"ard", ard},
        {"lengthscales", case_json.at("lengthscales_init")},
        {"noise_variance", case_json.at("noise_variance_init")},
        {"n_rows", n_rows},
        {"n_cols", n_cols},
        {"x", x_packed},
        {"y", y},
        {"xs_n_rows", xs_n_rows},
        {"xs_n_cols", xs_n_cols},
        {"xs", xs_packed},
        {"start_n", start_n},
        {"steps", steps},
    };
}

json run_time(const json& case_json) {
    const std::string name = case_json.at("name").get<std::string>();
    const bool ard = case_json.at("ard").get<bool>();
    const int n_rows = case_json.at("n_rows").get<int>();
    const int n_cols = case_json.at("n_cols").get<int>();
    const auto x_packed = case_json.at("x").get<std::vector<double>>();
    const auto y = case_json.at("y").get<std::vector<double>>();
    if (n_rows < 2) {
        return na_row(name, "n_rows must be at least 2");
    }

    const std::string cov =
        ard ? "CovSum ( CovSEard, CovNoise)" : "CovSum ( CovSEiso, CovNoise)";
    libgp::GaussianProcess probe(static_cast<size_t>(n_cols), cov);
    const Eigen::VectorXd loghyper = loghyper_from_case(case_json, probe);
    const std::size_t warmup = warmup_count();
    const std::size_t reps = timed_reps(n_rows);
    std::vector<double> insert_samples;
    insert_samples.reserve(reps);
    for (std::size_t i = 0; i < warmup + reps; ++i) {
        auto gp = std::make_unique<libgp::GaussianProcess>(static_cast<size_t>(n_cols), cov);
        gp->covf().set_loghyper(loghyper);
        add_point(*gp, x_packed, y, n_rows, n_cols, 0);
        add_point(*gp, x_packed, y, n_rows, n_cols, 1);
        const auto start = std::chrono::steady_clock::now();
        for (int index = 2; index < n_rows; ++index) {
            add_point(*gp, x_packed, y, n_rows, n_cols, index);
        }
        const double dt =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (i >= warmup) {
            insert_samples.push_back(dt);
        }
    }
    const auto [lo, hi] = min_max(insert_samples);
    return {
        {"lib", "libgp"},
        {"name", name},
        {"status", "ok"},
        {"factor_s", median(insert_samples)},
        {"factor_min_s", lo},
        {"factor_max_s", hi},
        {"eval_s", nullptr},
        {"predict_s", nullptr},
        {"joint_evals", nullptr},
        {"peak_rss_bytes", peak_rss_bytes()},
        {"warmup", warmup},
        {"reps", reps},
        {"note", "add_pattern from n=2 to n; first two points untimed"},
    };
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: libgp-online {golden|time} CASE.json\n";
        return 2;
    }
    const std::string mode = argv[1];
    std::ifstream in(argv[2]);
    if (!in) {
        std::cerr << "failed to open " << argv[2] << "\n";
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
        if (mode == "golden") {
            row = run_golden(case_json);
        } else if (mode == "time") {
            row = run_time(case_json);
        } else {
            std::cerr << "mode must be golden or time\n";
            return 2;
        }
    } catch (const std::exception& e) {
        const std::string name =
            case_json.contains("name") ? case_json["name"].get<std::string>() : std::string("unknown");
        if (mode == "time") {
            row = na_row(name, e.what());
        } else {
            std::cerr << e.what() << "\n";
            return 1;
        }
    }
    std::cout << row.dump() << "\n";
    return 0;
}
