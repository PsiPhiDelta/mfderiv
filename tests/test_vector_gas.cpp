// Massless quarks with a repulsive vector interaction. The internal variable is the effective
// chemical potential m, which maximizes Phi, and all derivatives of the pressure follow in closed form
// from the implicit function m(T, mu) defined by m + 2 G_V n(T, m) = mu.

#include <cmath>

#include <mfderiv/mfderiv.hpp>

#include "check.hpp"

namespace {

const double pi = 3.14159265358979323846;
const double g = 12.0;    // degeneracy
const double GV = 5e-6;   // MeV^-2

struct VectorGas {
    template <class S>
    S operator()(const mfd::VectorX<S>& x, const mfd::VectorX<S>& chi) const {
        const S& T = x(0);
        const S& mu = x(1);
        const S& m = chi(0);
        const S P = g * (7.0 * pi * pi * T * T * T * T / 360.0 + m * m * T * T / 12.0 + m * m * m * m / (24.0 * pi * pi));
        return -P - (mu - m) * (mu - m) / (4.0 * GV);
    }
};

// the same at T = 0, with x = (mu)
struct VectorGasT0 {
    template <class S>
    S operator()(const mfd::VectorX<S>& x, const mfd::VectorX<S>& chi) const {
        const S& m = chi(0);
        return -g * m * m * m * m / (24.0 * pi * pi) - (x(0) - m) * (x(0) - m) / (4.0 * GV);
    }
};

struct Exact {
    double m, P, s, n, c20, c11, c02, c30, c21, c12, c03, c04;
};

Exact exact(double T, double mu) {
    double m = mu;  // m + 2 GV n(T, m) = mu
    for (int i = 0; i < 100; ++i) {
        const double F = m + 2 * GV * g * (m * T * T / 6 + m * m * m / (6 * pi * pi)) - mu;
        const double dF = 1 + 2 * GV * g * (T * T / 6 + m * m / (2 * pi * pi));
        m -= F / dF;
    }
    Exact e{};
    e.m = m;
    const double n = g * (m * T * T / 6 + m * m * m / (6 * pi * pi));
    const double s = g * (7 * pi * pi * T * T * T / 90 + m * m * T / 6);
    const double n_m = g * (T * T / 6 + m * m / (2 * pi * pi)), n_mm = g * m / (pi * pi), n_mmm = g / (pi * pi);
    const double n_T = g * m * T / 3, n_TT = g * m / 3, n_mT = g * T / 3;
    const double s_T = g * (7 * pi * pi * T * T / 30 + m * m / 6), s_TT = g * 7 * pi * pi * T / 15;
    const double s_m = n_T, s_Tm = g * m / 3, s_mm = g * T / 3;
    const double k = 1 + 2 * GV * n_m;
    const double m_mu = 1 / k, m_T = -2 * GV * n_T / k;
    const double m_mumu = -2 * GV * n_mm * m_mu * m_mu * m_mu;
    const double m_muT = -2 * GV * (n_mT + n_mm * m_T) / (k * k);
    const double k_T = 2 * GV * (n_mT + n_mm * m_T);
    const double m_TT = -2 * GV * ((n_TT + n_mT * m_T) * k - n_T * k_T) / (k * k);
    const double m_mumumu = -2 * GV * (n_mmm * std::pow(m_mu, 4) + 3 * n_mm * m_mu * m_mu * m_mumu);
    e.P = g * (7 * pi * pi * std::pow(T, 4) / 360 + m * m * T * T / 12 + std::pow(m, 4) / (24 * pi * pi)) +
          (mu - m) * (mu - m) / (4 * GV);
    e.s = s;
    e.n = n;
    e.c20 = s_T + s_m * m_T;
    e.c11 = n_T * m_mu;
    e.c02 = n_m * m_mu;
    e.c30 = s_TT + s_Tm * m_T + (s_Tm + s_mm * m_T) * m_T + s_m * m_TT;
    e.c21 = (n_TT + n_mT * m_T) * m_mu + n_T * m_muT;
    e.c12 = (n_mT + n_mm * m_T) * m_mu + n_m * m_muT;
    e.c03 = n_mm * m_mu * m_mu + n_m * m_mumu;
    e.c04 = n_mmm * std::pow(m_mu, 3) + 3 * n_mm * m_mu * m_mumu + n_m * m_mumumu;
    return e;
}

void compare(const mfd::PressureDerivatives& r, const Exact& e, const std::string& tag) {
    check::close(r.chi(0), e.m, 1e-13, tag + " m");
    check::close(r.P, e.P, 1e-13, tag + " P");
    check::close(r.dP(0), e.s, 1e-13, tag + " s");
    check::close(r.dP(1), e.n, 1e-13, tag + " n");
    check::close(r.d2P(0, 0), e.c20, 1e-12, tag + " c20");
    check::close(r.d2P(0, 1), e.c11, 1e-12, tag + " c11");
    check::close(r.d2P(1, 0), e.c11, 1e-12, tag + " c11 symmetric");
    check::close(r.d2P(1, 1), e.c02, 1e-12, tag + " c02");
    check::close(r.d3P[0](0, 0), e.c30, 1e-10, tag + " c30");
    check::close(r.d3P[0](0, 1), e.c21, 1e-10, tag + " c21");
    check::close(r.d3P[1](0, 1), e.c12, 1e-10, tag + " c12");
    check::close(r.d3P[1](1, 1), e.c03, 1e-10, tag + " c03");
}

}  // namespace

int main() {
    const VectorGas phi;
    for (auto [T, mu] : {std::pair{40.0, 400.0}, {100.0, 200.0}, {5.0, 600.0}}) {
        const std::string tag = "T=" + std::to_string(int(T)) + " mu=" + std::to_string(int(mu));
        const Exact e = exact(T, mu);
        mfd::VectorXd x(2), chi0(1);
        x << T, mu;
        chi0 << 0.8 * mu;
        const auto easy = mfd::derivatives(phi, x, chi0, 3);
        compare(easy, e, "easy " + tag);
        const auto closed = mfd::closed_form_derivatives(phi, x, easy.chi, 3);
        compare(closed, e, "closed " + tag);
        check::close(closed.S(0, 1), 1 / (1 + 2 * GV * g * (T * T / 6 + e.m * e.m / (2 * pi * pi))), 1e-13,
                     tag + " dm/dmu");
        check::close(mfd::derivative<4>(phi, x, easy.chi, {1, 1, 1, 1}), e.c04, 1e-8, tag + " c04");
        // the naive second derivative misses the response of m
        check::truth(std::abs(closed.d2P_naive(1, 1) - e.c02) > 1e-3 * e.c02, tag + " naive differs");
    }

    // T = 0, x = (mu): c_s^2 = n / (mu dn/dmu)
    const VectorGasT0 phi0;
    for (double mu : {300.0, 450.0, 800.0}) {
        const Exact e = exact(0.0, mu);
        mfd::VectorXd x(1), chi0(1);
        x << mu;
        chi0 << mu;
        const auto r = mfd::derivatives(phi0, x, chi0, 2);
        check::close(r.dP(0), e.n, 1e-13, "T=0 n");
        check::close(r.d2P(0, 0), e.c02, 1e-12, "T=0 c02");
        check::close(mfd::sound_speed2(x, r.dP, r.d2P), e.n / (mu * e.c02), 1e-12, "T=0 c_s^2");
    }
    return check::report("vector_gas");
}
