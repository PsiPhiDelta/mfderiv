#pragma once
// Minimal test harness: each test is an executable that returns nonzero on failure.

#include <cmath>
#include <cstdio>
#include <string>

namespace check {

inline int checks = 0;
inline int failures = 0;

inline void close(double value, double expected, double rtol, const std::string& what, double atol = 0.0) {
    ++checks;
    const double err = std::abs(value - expected);
    if (!(err <= atol + rtol * std::abs(expected))) {
        ++failures;
        std::printf("FAIL %s: %.17g, expected %.17g (relative error %.3g)\n", what.c_str(), value, expected,
                    err / std::abs(expected));
    }
}

inline void truth(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

inline int report(const char* name) {
    std::printf("%s: %d checks, %d failures\n", name, checks, failures);
    return failures ? 1 : 0;
}

}  // namespace check
