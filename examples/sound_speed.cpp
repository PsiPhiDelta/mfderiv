// Equation of state and speed of sound of quarks with a repulsive vector interaction, at T = 0 and at
// finite temperature. The vector field is the only internal variable. The model is written once,
// generic in the scalar type, and all derivatives follow from automatic differentiation.

#include <cstdio>

#include <mfderiv/mfderiv.hpp>

const double pi = 3.14159265358979323846;

struct VectorGas {
    double GV = 5e-6;  // MeV^-2
    template <class S>
    S operator()(const mfd::VectorX<S>& x, const mfd::VectorX<S>& chi) const {
        const S& T = x(0);
        const S& mu = x(1);
        const S& m = chi(0);  // effective chemical potential mu - 2 G_V n
        const S P = 12.0 * (7.0 * pi * pi * T * T * T * T / 360.0 + m * m * T * T / 12.0 + m * m * m * m / (24.0 * pi * pi));
        return -P - (mu - m) * (mu - m) / (4.0 * GV);
    }
};

int main() {
    const VectorGas phi;
    const double hbc3 = 197.3269804 * 197.3269804 * 197.3269804;  // MeV^3 fm^3
    for (double T : {0.0, 50.0}) {
        std::printf("T = %g MeV\n%8s %12s %14s %14s %10s\n", T, "mu", "n [fm^-3]", "P [MeV/fm^3]", "e [MeV/fm^3]",
                    "c_s^2");
        mfd::VectorXd chi(1);
        chi << 300.0;
        for (double mu = 300.0; mu <= 600.0; mu += 50.0) {
            mfd::VectorXd x(2);
            x << T, mu;
            const auto r = mfd::derivatives(phi, x, chi, 2);  // P, (s, n), susceptibilities
            chi = r.chi;                                      // start the next point from this solution
            // at T = 0 the entropy vanishes and the T direction drops out of c_s^2
            const mfd::VectorXd y = T > 0 ? x : x.tail(1);
            const mfd::VectorXd sigma = T > 0 ? r.dP : r.dP.tail(1);
            const mfd::MatrixXd susc = T > 0 ? r.d2P : r.d2P.bottomRightCorner(1, 1);
            std::printf("%8.1f %12.5f %14.4f %14.4f %10.6f\n", mu, r.dP(1) / hbc3, r.P / hbc3,
                        mfd::energy_density(x, r.P, r.dP) / hbc3, mfd::sound_speed2(y, sigma, susc));
        }
    }
}
