// Response functions: the conformal free gas, process derivatives, Schur complements.

#include <cmath>

#include <mfderiv/mfderiv.hpp>

#include "check.hpp"

namespace {

const double pi = 3.14159265358979323846;

// massless free quarks, no internal variables
struct FreeGas {
    template <class S>
    S operator()(const mfd::VectorX<S>& x, const mfd::VectorX<S>&) const {
        const S& T = x(0);
        const S& mu = x(1);
        return -12.0 * (7.0 * pi * pi * T * T * T * T / 360.0 + mu * mu * T * T / 12.0 + mu * mu * mu * mu / (24.0 * pi * pi));
    }
};

struct VectorGas {
    template <class S>
    S operator()(const mfd::VectorX<S>& x, const mfd::VectorX<S>& chi) const {
        const S& T = x(0);
        const S& m = chi(0);
        const S P = 12.0 * (7.0 * pi * pi * T * T * T * T / 360.0 + m * m * T * T / 12.0 + m * m * m * m / (24.0 * pi * pi));
        return -P - (x(1) - m) * (x(1) - m) / (4.0 * 5e-6);
    }
};

}  // namespace

int main() {
    // conformal gas: c_s^2 = 1/3, Gamma = 4/3, c_V from the closed form
    for (auto [T, mu] : {std::pair{100.0, 0.0}, {50.0, 300.0}, {1.0, 500.0}}) {
        const std::string tag = "T=" + std::to_string(int(T)) + " mu=" + std::to_string(int(mu));
        mfd::VectorXd x(2);
        x << T, mu;
        const auto r = mfd::derivatives(FreeGas{}, x, mfd::VectorXd(0), 2);
        check::close(mfd::sound_speed2(x, r.dP, r.d2P), 1.0 / 3.0, 1e-13, tag + " c_s^2");
        check::close(mfd::adiabatic_index(x, r.P, r.dP, r.d2P), 4.0 / 3.0, 1e-13, tag + " Gamma");
        check::close(mfd::energy_density(x, r.P, r.dP), 3 * r.P, 1e-13, tag + " epsilon = 3P");
        const double c20 = 12 * (7 * pi * pi * T * T / 30 + mu * mu / 6), c11 = 12 * mu * T / 3,
                     c02 = 12 * (T * T / 6 + mu * mu / (2 * pi * pi));
        check::close(mfd::heat_capacity_V(x, r.d2P), T * (c20 - c11 * c11 / c02), 1e-12, tag + " c_V");
    }

    // c_s^2 and the hydrodynamic derivatives as process derivatives
    {
        mfd::VectorXd x(2), chi0(1);
        x << 40.0, 400.0;
        chi0 << 350.0;
        const auto r = mfd::derivatives(VectorGas{}, x, chi0, 2);
        const double s = r.dP(0), n = r.dP(1);
        const mfd::VectorXd grad_eps = r.d2P * x;                               // d epsilon = x . d sigma
        const mfd::MatrixXd grad_sn = (n * r.d2P.col(0) - s * r.d2P.col(1)) / (n * n);  // d(s/n)
        check::close(mfd::sound_speed2(x, r.dP, r.d2P), mfd::process_derivative(r.dP, grad_eps, grad_sn), 1e-12,
                     "c_s^2 as process derivative");
        const mfd::VectorXd hydro = mfd::hydro_derivatives(x, r.dP, r.d2P);
        check::close(hydro(0), mfd::process_derivative(r.dP, grad_eps, r.d2P.col(1)), 1e-12, "dP/deps at fixed n");
        check::close(hydro(1), mfd::process_derivative(r.dP, r.d2P.col(1), grad_eps), 1e-12, "dP/dn at fixed eps");
        const double cs2 = mfd::sound_speed2(x, r.dP, r.d2P);
        check::truth(cs2 > 1.0 / 3.0 && cs2 < 1.0, "repulsion stiffens, causal");
    }

    // Schur complement for a fixed density
    {
        mfd::MatrixXd chi(3, 3);
        chi << 4, 1, 0.5, 1, 3, 0.2, 0.5, 0.2, 2;
        const mfd::MatrixXd c = mfd::constrain(chi, {1});
        check::close(c(0, 0), 4 - 1 * 1 / 3.0, 1e-15, "constrain 00");
        check::close(c(0, 1), 0.5 - 1 * 0.2 / 3.0, 1e-15, "constrain 02");
        check::close(c(1, 1), 2 - 0.2 * 0.2 / 3.0, 1e-15, "constrain 22");
    }
    return check::report("response");
}
