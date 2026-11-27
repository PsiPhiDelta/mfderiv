// Chiral mean-field (CMF) model of symmetric nuclear matter at T = 0, compared with the CMF++ module of MUSES.
//
// Nucleons couple to the scalar fields sigma and zeta and to the vector field omega. The dilaton is frozen at its
// vacuum value chi_0, the vector self-interaction is the C4 form restricted to omega, and the phi, rho and delta
// fields vanish. The parameters and the vacuum subtraction are those of the default C4 configuration of CMF++,
// and g_sigma, g_zeta are the nucleon couplings it derives from the baryon vacuum masses.
//
// usage: showcase_cmf <CMF++ output.csv> <out.csv>   at the points of a CMF++ run, started from its solutions

#include <cmath>
#include <cstdio>
#include <string>

#include <mfderiv/mfderiv.hpp>

#include "common.hpp"

using showcase::fermi_gas;
using showcase::hbc3;

struct CMF {
    double k0 = 2.3732188, k1 = 1.39999998, k2 = -5.54911336, k3 = -2.65241888;
    double chi0 = 401.933763, d = 0.0606060606;  // dilaton vacuum value in MeV, d = 2/33 from the QCD beta function
    double f_pi = 93.3000031, f_K = 122.0, m_pi = 139.0, m_K = 498.0;
    double m0 = 150.0, g_sigma = -9.83449216548, g_zeta = 1.22291627408;
    double g_omega = 11.9, g4 = 38.9, m_omega = 780.562988;
    double gamma = 4.0;  // spin and isospin
    double sigma0 = -f_pi, zeta0 = (f_pi - 2.0 * f_K) / std::sqrt(2.0);  // vacuum, the reference point of U

    // scalar potential, U = -L_scalar - L_symmetry breaking
    template <class S>
    S U(const S& s, const S& z) const {
        using std::log;
        const S r2 = s * s + z * z;
        return 0.5 * k0 * chi0 * chi0 * r2 - k1 * r2 * r2 - k2 * (0.5 * s * s * s * s + z * z * z * z) -
               k3 * chi0 * s * s * z - d / 3.0 * std::pow(chi0, 4) * log(s * s * z / (sigma0 * sigma0 * zeta0)) +
               m_pi * m_pi * f_pi * s + (std::sqrt(2.0) * m_K * m_K * f_K - m_pi * m_pi * f_pi / std::sqrt(2.0)) * z;
    }

    // x = (mu_B), chi = (sigma, zeta, omega) in MeV
    template <class S>
    S operator()(const mfd::VectorX<S>& x, const mfd::VectorX<S>& chi) const {
        const S& s = chi(0);
        const S& z = chi(1);
        const S& w = chi(2);
        const S m_eff = m0 + g_sigma * s + g_zeta * z;
        const S mu_eff = x(0) - g_omega * w;
        return U(s, z) - U(sigma0, zeta0) - 0.5 * m_omega * m_omega * w * w - g4 * w * w * w * w -
               gamma / 2.0 * fermi_gas(mu_eff, m_eff);
    }
};

// P, n_B, epsilon in MeV/fm^3 and fm^-3, c_s^2, fields and nucleon effective mass at one mu_B
std::vector<double> point(const CMF& cmf, double muB, mfd::VectorXd& chi) {
    mfd::VectorXd x(1);
    x << muB;
    const auto r = mfd::derivatives(cmf, x, chi, 2);
    chi = r.chi;
    const double n = r.dP(0), cs2 = n / (muB * r.d2P(0, 0));
    return {muB, r.P / hbc3, n / hbc3, (muB * n - r.P) / hbc3, cs2, chi(0), chi(1), chi(2),
            cmf.m0 + cmf.g_sigma * chi(0) + cmf.g_zeta * chi(1)};
}

const std::vector<std::string> columns = {"muB", "P", "nB", "e", "cs2", "sigma", "zeta", "omega", "m_eff"};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <CMF++ output.csv> <out.csv>\n", argv[0]);
        return 2;
    }
    const CMF cmf;
    const showcase::Table ref = showcase::read_csv(argv[1]);
    std::vector<std::string> names = columns;
    for (const char* c : {"P_ref", "nB_ref", "e_ref", "sigma_ref", "zeta_ref", "omega_ref"}) names.push_back(c);
    showcase::Writer out(argv[2], names);
    showcase::Deviations dP, dn, de, df;
    int used = 0, failed = 0;
    for (std::size_t i = 0; i < ref.rows.size(); ++i) {
        if (ref(i, "nB") < 1e-6) continue;  // the vacuum, where c_s^2 is not defined
        ++used;
        mfd::VectorXd chi(3);
        chi << ref(i, "sigma_mean_field"), ref(i, "zeta_mean_field"), ref(i, "omega_mean_field");
        std::vector<double> row;
        try {
            row = point(cmf, ref(i, "muB"), chi);
        } catch (const std::exception& e) {
            std::printf("mu_B = %g: %s\n", ref(i, "muB"), e.what());
            ++failed;
            continue;
        }
        const std::vector<double> r = {ref(i, "pressure"),         ref(i, "nB"),
                                       ref(i, "energy_density"),   ref(i, "sigma_mean_field"),
                                       ref(i, "zeta_mean_field"),  ref(i, "omega_mean_field")};
        dP.add(std::abs(row[1] - r[0]));
        dn.add(std::abs(row[2] / r[1] - 1.0));
        de.add(std::abs(row[3] / r[2] - 1.0));
        df.add(std::max({std::abs(row[5] - r[3]), std::abs(row[6] - r[4]), std::abs(row[7] - r[5])}));
        row.insert(row.end(), r.begin(), r.end());
        out.row(row);
    }
    std::printf("%d CMF++ points with nucleons, %d without convergence here\n", used, failed);
    std::printf("deviation from CMF++ (median, max): P %.1e, %.1e MeV/fm^3 | n_B %.1e, %.1e | e %.1e, %.1e (relative)"
                " | fields %.1e, %.1e MeV\n",
                dP.median(), dP.max(), dn.median(), dn.max(), de.median(), de.max(), df.median(), df.max());
    return used > failed ? 0 : 1;
}
