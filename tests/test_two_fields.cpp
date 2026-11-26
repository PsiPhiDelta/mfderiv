// A model with two internal variables of opposite curvature, a scalar field sigma (minimum of Phi)
// and a vector field omega (maximum), compared with finite differences of the pressure.

#include <cmath>

#include <mfderiv/mfderiv.hpp>

#include "check.hpp"

namespace {

const double pi = 3.14159265358979323846;

struct SigmaOmega {
    double g = 12.0, G = 2e-5, GV = 5e-6, m0 = 300.0, L2 = 600.0 * 600.0;
    template <class S>
    S operator()(const mfd::VectorX<S>& x, const mfd::VectorX<S>& chi) const {
        using std::exp;
        const S& T = x(0);
        const S mt = x(1) - chi(1);
        const S M = m0 + chi(0);
        const S D = T * T + mt * mt + L2;
        const S Pq = g * (7.0 * pi * pi * T * T * T * T / 360.0 + mt * mt * T * T / 12.0 + mt * mt * mt * mt / (24.0 * pi * pi)) *
                     exp(-M * M / D);
        return chi(0) * chi(0) / (4.0 * G) - chi(1) * chi(1) / (4.0 * GV) - Pq;
    }
};

// second derivatives of P by central differences, Richardson-extrapolated
mfd::MatrixXd fd_hessian(const SigmaOmega& phi, const mfd::VectorXd& x, const mfd::VectorXd& chi, double h) {
    mfd::NewtonOptions opt;
    opt.tol = 1e-15;
    auto P = [&](const mfd::VectorXd& y) { return mfd::pressure(phi, y, chi, opt); };
    auto second = [&](int a, int b, double e) {
        mfd::VectorXd ea = mfd::VectorXd::Unit(2, a) * e, eb = mfd::VectorXd::Unit(2, b) * e;
        if (a == b) return (P(x + ea) - 2 * P(x) + P(x - ea)) / (e * e);
        return (P(x + ea + eb) - P(x + ea - eb) - P(x - ea + eb) + P(x - ea - eb)) / (4 * e * e);
    };
    mfd::MatrixXd H(2, 2);
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b) H(a, b) = (4 * second(a, b, h / 2) - second(a, b, h)) / 3;
    return H;
}

}  // namespace

int main() {
    const SigmaOmega phi;
    for (auto [T, mu] : {std::pair{100.0, 300.0}, {30.0, 500.0}, {150.0, 50.0}}) {
        const std::string tag = "T=" + std::to_string(int(T)) + " mu=" + std::to_string(int(mu));
        mfd::VectorXd x(2), chi0(2);
        x << T, mu;
        chi0 << -50.0, 20.0;
        const auto easy = mfd::derivatives(phi, x, chi0, 3);
        const auto closed = mfd::closed_form_derivatives(phi, x, easy.chi, 3);

        // stationarity: minimum in sigma, maximum in omega
        mfd::VectorXd g;
        mfd::MatrixXd H;
        double v;
        mfd::detail::hessian(mfd::detail::Joined<SigmaOmega>{phi, 2}, mfd::detail::concat<double>(x, easy.chi), 2, v, g,
                             H);
        check::truth(g.cwiseAbs().maxCoeff() < 1e-6, tag + " gap equations solved");
        check::truth(H(0, 0) > 0 && H(1, 1) < 0, tag + " sigma minimum, omega maximum");

        check::close(easy.P, closed.P, 1e-14, tag + " P");
        for (int a = 0; a < 2; ++a) {
            check::close(easy.dP(a), closed.dP(a), 1e-13, tag + " dP");
            for (int b = 0; b < 2; ++b) {
                check::close(easy.d2P(a, b), closed.d2P(a, b), 1e-11, tag + " d2P easy vs closed");
                for (int c = 0; c < 2; ++c)
                    check::close(easy.d3P[a](b, c), closed.d3P[a](b, c), 1e-9, tag + " d3P easy vs closed");
            }
        }
        const mfd::MatrixXd fd = fd_hessian(phi, x, easy.chi, 2.0);
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) check::close(easy.d2P(a, b), fd(a, b), 1e-7, tag + " d2P vs finite differences");

        // the response of the solution, S = dchibar/dx, against finite differences of chibar
        for (int a = 0; a < 2; ++a) {
            const double h = 1e-3;
            mfd::VectorXd e = mfd::VectorXd::Unit(2, a) * h;
            const mfd::VectorXd dchi = (mfd::gap(phi, mfd::VectorXd(x + e), easy.chi) -
                                        mfd::gap(phi, mfd::VectorXd(x - e), easy.chi)) / (2 * h);
            for (int i = 0; i < 2; ++i) check::close(closed.S(i, a), dchi(i), 1e-6, tag + " S", 1e-9);
        }
    }
    return check::report("two_fields");
}
