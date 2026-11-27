#pragma once
// Pressure derivatives from closed formulas at a solution chibar of the gap equations (the
// envelope theorem, the Schur complement and the tangent contraction), up to third order. They need
// only partial derivatives of Phi at one point and the linear response S = dchibar/dx. For higher
// orders use derivative<K>() of derivatives.hpp. The formulas hold only at a solution, so chibar is
// checked: the Newton step from it must be below tol relative to (1 + |chibar|).

#include <array>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "core.hpp"

namespace mfd {

namespace detail {

// {f, f', f'', f'''} of s -> f(z + s h) at s = 0
template <class F>
std::array<double, 4> directional3(const F& f, const VectorXd& z, const VectorXd& h) {
    using D1 = autodiff::dual;
    using D2 = autodiff::detail::Dual<D1, D1>;
    using D3 = autodiff::detail::Dual<D2, D2>;
    D3 s{};
    s.val.val.grad = 1.0;
    s.val.grad.val = 1.0;
    s.grad.val.val = 1.0;
    VectorX<D3> w(z.size());
    for (Index k = 0; k < z.size(); ++k) w(k) = ad::lift<D3>(z(k)) + s * h(k);
    const D3 r = f(w);
    return {r.val.val.val, r.val.val.grad, r.val.grad.grad, r.grad.grad.grad};
}

}  // namespace detail

template <class Model>
PressureDerivatives closed_form_derivatives(const Model& phi, const VectorXd& x, const VectorXd& chibar,
                                            int order = 2, double tol = 1e-8) {
    if (order < 1 || order > 3)
        throw std::invalid_argument("mfd::closed_form_derivatives: order must be 1, 2 or 3");
    const Index d = x.size(), N = chibar.size();
    const detail::Joined<Model> f{phi, d};
    const VectorXd z = detail::concat<double>(x, chibar);
    double v;
    VectorXd g;
    MatrixXd H;
    detail::hessian(f, z, 0, v, g, H);
    if (N) {
        const double step = H.bottomRightCorner(N, N).partialPivLu().solve(g.tail(N)).norm();
        if (!(step <= tol * (1.0 + chibar.norm())))
            throw std::invalid_argument("mfd::closed_form_derivatives: chibar does not solve the gap equations "
                                        "(Newton step " + std::to_string(step) + ")");
    }

    PressureDerivatives r;
    r.P = -v;
    r.dP = -g.head(d);                                       // envelope theorem
    r.chi = chibar;
    r.d2P_naive = -H.topLeftCorner(d, d);
    if (N) {
        r.S = -H.bottomRightCorner(N, N).partialPivLu().solve(H.bottomLeftCorner(N, d));
        r.d2P = -(H.topLeftCorner(d, d) + H.topRightCorner(d, N) * r.S);  // Schur complement
    } else {
        r.S = MatrixXd(0, d);
        r.d2P = r.d2P_naive;
    }
    if (order < 3) return r;

    // tangent contraction, d3Omega[a, b, c] = Phi'''[t_a, t_b, t_c] with t = (1, S), from directional
    // third derivatives along integer combinations of the tangents (polarization of a cubic form)
    MatrixXd t(d + N, d);
    t.topRows(d) = MatrixXd::Identity(d, d);
    t.bottomRows(N) = r.S;
    std::map<std::vector<int>, double> cache;
    auto cube = [&](std::vector<int> c) {
        auto it = cache.find(c);
        if (it != cache.end()) return it->second;
        VectorXd h = VectorXd::Zero(d + N);
        for (Index a = 0; a < d; ++a) h += c[a] * t.col(a);
        const double val = detail::directional3(f, z, h)[3];
        cache.emplace(c, val);
        return val;
    };
    auto unit = [&](std::initializer_list<Index> idx) {
        std::vector<int> c(d, 0);
        for (Index a : idx) ++c[a];
        return c;
    };
    r.d3P.assign(d, MatrixXd(d, d));
    for (Index a = 0; a < d; ++a)
        for (Index b = a; b < d; ++b)
            for (Index c = b; c < d; ++c) {
                const double T = (cube(unit({a, b, c})) - cube(unit({a, b})) - cube(unit({a, c})) -
                                  cube(unit({b, c})) + cube(unit({a})) + cube(unit({b})) + cube(unit({c}))) / 6.0;
                for (auto [i, j, k] : {std::array<Index, 3>{a, b, c}, {a, c, b}, {b, a, c}, {b, c, a}, {c, a, b},
                                       {c, b, a}})
                    r.d3P[i](j, k) = -T;
            }
    return r;
}

}  // namespace mfd
