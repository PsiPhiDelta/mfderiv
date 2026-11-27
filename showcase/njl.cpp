// Three-flavor NJL model of the MUSES NJL module in normal quark matter at T = 0, compared with the module.
//
// The module uses the parameters of Rehberg, Klevansky and Huefner and the minimal RG-consistent scheme. Without
// diquark condensates the RG counterterm vanishes and the medium part, which runs up to Lambda_RG = 20 Lambda,
// is the full Fermi sea. Only the Dirac sea is cut off at Lambda, and at T = 0 both are elementary functions.
// The internal variables are the condensates phi_f and the effective chemical potential mu_V = mu - 2 G_V n of
// the flavor-singlet vector interaction, and in neutral matter also the charge chemical potential mu_Q, with
// electrons and muons in beta equilibrium. mu is the quark chemical potential and n the quark number density.
//
// usage: showcase_njl <module output.csv> <out.csv>   symmetric quark matter at the points of a module run,
//                                                     started from its solutions
//        showcase_njl --neutral <out.csv>             neutral matter in beta equilibrium, followed in mu

#include <array>
#include <cmath>
#include <cstdio>
#include <string>

#include <mfderiv/mfderiv.hpp>

#include "common.hpp"

using showcase::fermi_gas;
using showcase::hbc3;
using showcase::pi;

// Dirac sea of one flavor and three colors with a sharp cutoff
template <class S>
S dirac_sea(const S& M, double L) {
    using std::log;
    using std::sqrt;
    const S e = sqrt(L * L + M * M);
    return -3.0 / (8.0 * pi * pi) * (L * e * (2.0 * L * L + M * M) - M * M * M * M * log((L + e) / M));
}

struct NJL {
    double Lambda = 602.3;  // MeV
    double G = 1.835 / (Lambda * Lambda), K = 12.36 / std::pow(Lambda, 5), GV = 0.7 * G;
    std::array<double, 3> m = {5.5, 5.5, 140.7};
    double m_e = 0.511, m_mu = 105.66;
    bool neutral = false;  // charge neutral matter with leptons, otherwise mu_Q = 0 without leptons
    double vacuum = 0.0;   // Phi of the vacuum, subtracted

    // condensates and Dirac sea, with chi(0..2) = phi_f / Lambda^2 in MeV
    template <class S>
    S sea(const mfd::VectorX<S>& chi, std::array<S, 3>& M) const {
        const double L2 = Lambda * Lambda;
        const S pu = L2 * chi(0), pd = L2 * chi(1), ps = L2 * chi(2);
        M = {m[0] - 4.0 * G * pu + 2.0 * K * pd * ps, m[1] - 4.0 * G * pd + 2.0 * K * pu * ps,
             m[2] - 4.0 * G * ps + 2.0 * K * pu * pd};
        S om = 2.0 * G * (pu * pu + pd * pd + ps * ps) - 4.0 * K * pu * pd * ps;
        for (const S& Mf : M) om += dirac_sea(Mf, Lambda);
        return om;
    }

    // x = (mu), chi = (phi_u, phi_d, phi_s) / Lambda^2, mu_V and, in neutral matter, mu_Q
    template <class S>
    S operator()(const mfd::VectorX<S>& x, const mfd::VectorX<S>& chi) const {
        std::array<S, 3> M;
        S om = sea(chi, M);
        const S& mu = x(0);
        const S& muV = chi(3);
        const S muQ = neutral ? chi(4) : S(0.0);
        const std::array<S, 3> muf = {muV + 2.0 / 3.0 * muQ, muV - muQ / 3.0, muV - muQ / 3.0};
        for (int f = 0; f < 3; ++f) om -= 3.0 * fermi_gas(muf[f], M[f]);
        om -= (mu - muV) * (mu - muV) / (4.0 * GV);
        if (neutral) {
            const S muL = -muQ;  // leptons in beta equilibrium
            om -= fermi_gas(muL, S(m_e)) + fermi_gas(muL, S(m_mu));
        }
        return om - vacuum;
    }
};

// the vacuum, with the condensates as the only internal variables
struct Vacuum {
    const NJL& njl;
    template <class S>
    S operator()(const mfd::VectorX<S>&, const mfd::VectorX<S>& chi) const {
        std::array<S, 3> M;
        return njl.sea(chi, M);
    }
};

// P, n, epsilon in MeV/fm^3 and fm^-3, c_s^2, masses and internal variables at one mu
std::vector<double> point(const NJL& njl, double mu, mfd::VectorXd& chi) {
    mfd::VectorXd x(1);
    x << mu;
    const auto r = mfd::derivatives(njl, x, chi, 2);
    chi = r.chi;
    std::array<double, 3> M;
    njl.sea(r.chi, M);
    const double n = r.dP(0), cs2 = n / (mu * r.d2P(0, 0));
    return {mu, r.P / hbc3, n / hbc3, (mu * n - r.P) / hbc3, cs2, M[0], M[1], M[2], chi(3), njl.neutral ? chi(4) : 0.0};
}

const std::vector<std::string> columns = {"mu", "P", "n", "e", "cs2", "M_u", "M_d", "M_s", "mu_V", "mu_Q"};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <module output.csv> <out.csv> | --neutral <out.csv>\n", argv[0]);
        return 2;
    }
    NJL njl;
    const Vacuum vac{njl};
    mfd::VectorXd x0 = mfd::VectorXd::Zero(1), c0(3);
    c0 << -39.0, -39.0, -47.0;
    const mfd::VectorXd cv = mfd::newton(vac, x0, c0).chi;
    njl.vacuum = vac(x0, cv);
    std::array<double, 3> Mv;
    njl.sea(cv, Mv);
    std::printf("vacuum masses %.6f %.6f %.6f MeV\n", Mv[0], Mv[1], Mv[2]);

    if (std::string(argv[1]) == "--neutral") {
        njl.neutral = true;
        showcase::Writer out(argv[2], columns);
        mfd::VectorXd chi(5);
        chi << cv(0), cv(1), cv(2), 372.0, -5.0;  // start just above the quark onset
        for (double mu = 374.0; mu <= 800.0 + 1e-9; mu += 2.0) out.row(point(njl, mu, chi));
        return 0;
    }

    const showcase::Table ref = showcase::read_csv(argv[1]);
    std::vector<std::string> names = columns;
    for (const char* c : {"P_ref", "n_ref", "e_ref", "M_u_ref", "M_d_ref", "M_s_ref", "mu_V_ref"}) names.push_back(c);
    showcase::Writer out(argv[2], names);
    const double L2 = njl.Lambda * njl.Lambda;
    showcase::Deviations dP, dn, de, dM, dV;
    std::vector<double> offset;
    int used = 0, failed = 0;
    for (std::size_t i = 0; i < ref.rows.size(); ++i) {
        // converged module points with quarks, where c_s^2 is defined
        if (ref(i, "success") != 0.0 || ref(i, "n") < 1e-6) continue;
        ++used;
        mfd::VectorXd chi(4);
        chi << ref(i, "phi_u") / L2, ref(i, "phi_d") / L2, ref(i, "phi_s") / L2, ref(i, "mu_tilde_V");
        std::vector<double> row;
        try {
            row = point(njl, ref(i, "mu"), chi);
        } catch (const std::exception& e) {
            std::printf("mu = %g: %s\n", ref(i, "mu"), e.what());
            ++failed;
            continue;
        }
        const std::vector<double> r = {-ref(i, "Omega"), ref(i, "n"),   ref(i, "energy_density"), ref(i, "M_u"),
                                       ref(i, "M_d"),    ref(i, "M_s"), ref(i, "mu_tilde_V")};
        offset.push_back(row[1] - r[0]);
        dn.add(std::abs(row[2] / r[1] - 1.0));
        de.add(std::abs(row[3] / r[2] - 1.0));
        dM.add(std::max({std::abs(row[5] - r[3]), std::abs(row[6] - r[4]), std::abs(row[7] - r[5])}));
        dV.add(std::abs(row[8] - r[6]));
        row.insert(row.end(), r.begin(), r.end());
        out.row(row);
    }
    // the module subtracts its own numerical vacuum, so P differs by a constant
    showcase::Deviations o;
    for (double v : offset) o.add(v);
    const double c = o.median();
    for (double v : offset) dP.add(std::abs(v - c));
    std::printf("%d module points with quarks, %d without convergence here\n", used, failed);
    std::printf("P - P_module = %.2e MeV/fm^3 (vacuum subtraction of the module), deviation from it (median, max) "
                "%.1e, %.1e\n", c, dP.median(), dP.max());
    std::printf("deviation from the module (median, max): n %.1e, %.1e | e %.1e, %.1e (relative) | masses %.1e, %.1e"
                " | mu_V %.1e, %.1e MeV\n",
                dn.median(), dn.max(), de.median(), de.max(), dM.median(), dM.max(), dV.median(), dV.max());
    return used > failed ? 0 : 1;
}
