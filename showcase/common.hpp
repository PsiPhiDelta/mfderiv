#pragma once
// Shared pieces of the showcase programs: the T = 0 Fermi gas and small CSV tables.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace showcase {

const double pi = 3.14159265358979323846;
const double hbc = 197.3269804;  // MeV fm
const double hbc3 = hbc * hbc * hbc;

// pressure of a free Fermi gas with two spin states at T = 0. The Fermi surface is resolved analytically, so
// the expression is smooth in mu and m away from the threshold mu = m.
template <class S>
S fermi_gas(const S& mu, const S& m) {
    using std::log;
    using std::sqrt;
    if (mu <= m) return S(0.0);
    const S k = sqrt(mu * mu - m * m);
    return (mu * k * (2.0 * mu * mu - 5.0 * m * m) + 3.0 * m * m * m * m * log((mu + k) / m)) / (24.0 * pi * pi);
}

struct Table {
    std::vector<std::string> names;
    std::vector<std::vector<double>> rows;

    int col(const std::string& name) const {
        for (int i = 0; i < int(names.size()); ++i)
            if (names[i] == name) return i;
        throw std::runtime_error("no column " + name);
    }
    double operator()(std::size_t row, const std::string& name) const { return rows[row][col(name)]; }
};

inline Table read_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    Table t;
    std::string line, cell;
    std::getline(in, line);
    for (std::stringstream ss(line); std::getline(ss, cell, ',');) t.names.push_back(cell);
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        for (std::stringstream ss(line); std::getline(ss, cell, ',');) {
            char* end = nullptr;
            const double v = std::strtod(cell.c_str(), &end);
            row.push_back(end != cell.c_str() ? v : std::nan(""));  // text columns, such as the phase, become NaN
        }
        t.rows.push_back(row);
    }
    return t;
}

// median and maximum of a set of deviations
struct Deviations {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    double median() const {
        std::vector<double> w = v;
        std::nth_element(w.begin(), w.begin() + w.size() / 2, w.end());
        return w.empty() ? 0.0 : w[w.size() / 2];
    }
    double max() const { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); }
};

class Writer {
public:
    Writer(const std::string& path, const std::vector<std::string>& names) : f_(std::fopen(path.c_str(), "w")) {
        if (!f_) throw std::runtime_error("cannot write " + path);
        for (std::size_t i = 0; i < names.size(); ++i) std::fprintf(f_, i ? ",%s" : "%s", names[i].c_str());
        std::fprintf(f_, "\n");
    }
    ~Writer() { std::fclose(f_); }
    void row(const std::vector<double>& values) {
        for (std::size_t i = 0; i < values.size(); ++i) std::fprintf(f_, i ? ",%.12g" : "%.12g", values[i]);
        std::fprintf(f_, "\n");
    }

private:
    std::FILE* f_;
};

}  // namespace showcase
