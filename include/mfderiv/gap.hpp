#pragma once
// Gap equations Phi_chi(x, chi) = 0 and the pressure P(x) = -Phi(x, chibar(x)).
//
// gap(phi, x, chi0) works for x of type double and of any nested autodiff dual type. For double it
// runs a Newton iteration. For a dual number it first solves the less nested problem (one layer
// peeled off), embeds that solution with zero outer tangent, and takes a single Newton step in the
// outer layer. Since the starting point is wrong only at first order in the outer tangent, this
// step makes it exact (implicit-function theorem). Hence pressure(phi, x, chi0) is an ordinary
// differentiable function of x for x of any nested dual type, and derivatives.hpp returns its
// derivatives of any order. The gap equations are
// stationarity conditions, not minimality conditions, so internal variables that maximize Phi
// (repulsive vector fields, chemical potentials fixed by neutrality) are treated alike.

#include <cmath>
#include <stdexcept>
#include <string>

#include "core.hpp"

namespace mfd {

struct NewtonOptions {
    double tol = 1e-12;   // relative step size at convergence
    int max_iter = 100;
    int max_halvings = 30;  // backtracking on the norm of the gap equations
};

struct NewtonResult {
    VectorXd chi;
    double residual = 0.0;  // max |Phi_chi|
    int iterations = 0;
    bool converged = false;
};

// Newton iteration for Phi_chi(x, chi) = 0 at fixed x, starting from chi0
template <class Model>
NewtonResult newton(const Model& phi, const VectorXd& x, const VectorXd& chi0, const NewtonOptions& opt = {}) {
    const Index d = x.size();
    const detail::Joined<Model> f{phi, d};
    NewtonResult res;
    VectorXd chi = chi0;
    if (chi.size() == 0) {
        res.chi = chi;
        res.converged = true;
        return res;
    }
    auto equations = [&](const VectorXd& c, VectorXd& g, MatrixXd& H) {
        double v;
        detail::hessian(f, detail::concat<double>(x, c), d, v, g, H);
    };
    VectorXd g;
    MatrixXd H;
    equations(chi, g, H);
    for (res.iterations = 0; res.iterations < opt.max_iter; ++res.iterations) {
        const VectorXd step = H.partialPivLu().solve(g);
        if (!step.allFinite()) break;
        double t = 1.0;
        VectorXd trial = chi - step, gt;
        MatrixXd Ht;
        equations(trial, gt, Ht);
        // At a solution, for instance when derivatives restart from a converged chi, rounding noise can let the
        // norm grow. A step below the tolerance is taken in full.
        const bool converging = step.norm() <= opt.tol * (1.0 + chi.norm());
        for (int k = 0; !converging && k < opt.max_halvings && !(gt.allFinite() && gt.norm() <= g.norm()); ++k) {
            t *= 0.5;
            trial = chi - t * step;
            equations(trial, gt, Ht);
        }
        chi = trial;
        g = gt;
        H = Ht;
        if (t * step.norm() <= opt.tol * (1.0 + chi.norm())) {
            res.converged = true;
            ++res.iterations;
            break;
        }
    }
    res.chi = chi;
    res.residual = g.size() ? g.cwiseAbs().maxCoeff() : 0.0;
    return res;
}

// chibar(x) with the derivatives of x carried by its dual type
template <class Model, class S>
VectorX<S> gap(const Model& phi, const VectorX<S>& x, const VectorXd& chi0, const NewtonOptions& opt = {}) {
    if constexpr (!ad::is_dual_v<S>) {
        const NewtonResult r = newton(phi, x, chi0, opt);
        if (!r.converged)
            throw std::runtime_error("mfd::gap: Newton iteration did not converge (residual " +
                                     std::to_string(r.residual) + ")");
        return r.chi;
    } else {
        if (chi0.size() == 0) return VectorX<S>(0);
        using I = ad::inner_t<S>;
        const Index d = x.size();
        const detail::Joined<Model> f{phi, d};
        const VectorX<I> x0 = ad::value(x);
        const VectorX<I> chi = gap(phi, x0, chi0, opt);
        // Hessian in the inner layer, gap equations in the outer layer. Their primal part vanishes,
        // so the outer tangent of the Hessian does not contribute to the Newton step.
        I v;
        VectorX<I> g0;
        MatrixX<I> H;
        detail::hessian(f, detail::concat<I>(x0, chi), d, v, g0, H);
        const VectorX<S> g = detail::gradient(f, detail::concat<S>(x, ad::lift<S>(chi)), d);
        const Eigen::PartialPivLU<MatrixXd> lu(detail::primal(H));
        const VectorX<I> dv = detail::solve(lu, H, ad::value(g));
        const VectorX<I> dt = detail::solve(lu, H, ad::tangent(g));
        VectorX<S> out(chi.size());
        for (Index i = 0; i < chi.size(); ++i) out(i) = ad::make<S>(chi(i) - dv(i), -dt(i));
        return out;
    }
}

// pressure P(x) = -Phi(x, chibar(x)), differentiable in x through any nested dual type
template <class Model, class S>
S pressure(const Model& phi, const VectorX<S>& x, const VectorXd& chi0, const NewtonOptions& opt = {}) {
    return -phi(x, gap(phi, x, chi0, opt));
}

}  // namespace mfd
