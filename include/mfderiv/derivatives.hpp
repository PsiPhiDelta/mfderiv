#pragma once
// Derivatives of the pressure through the implicit-function-aware gap solver of gap.hpp. The
// pressure is evaluated with nested dual numbers, one layer per derivative, and each layer of the
// gap solution costs one Newton step, so any order is available without derivatives of the
// solution itself.

#include <array>
#include <cstddef>

#include "core.hpp"
#include "gap.hpp"

namespace mfd {
namespace detail {

// set the tangent of layer `level` (1 = innermost) of a nested dual number to `value`
template <class S>
void seed(S& v, int level, double value) {
    if (level == ad::depth<S>()) {
        v.grad = ad::lift<ad::inner_t<S>>(value);
    } else if constexpr (ad::is_dual_v<ad::inner_t<S>>) {
        seed(v.val, level, value);
    }
}

// outermost tangent of every layer, the mixed derivative with respect to all seeded directions
template <class S>
double top(const S& v) {
    if constexpr (ad::is_dual_v<S>)
        return top(v.grad);
    else
        return v;
}

}  // namespace detail

// d^K P / dx_{idx[0]} ... dx_{idx[K-1]} for any order K
template <std::size_t K, class Model>
double derivative(const Model& phi, const VectorXd& x, const VectorXd& chi0, const std::array<Index, K>& idx,
                  const NewtonOptions& opt = {}) {
    using D = autodiff::HigherOrderDual<K, double>;
    VectorX<D> xd = ad::lift<D>(x);
    for (std::size_t l = 0; l < K; ++l) detail::seed(xd(idx[l]), int(l) + 1, 1.0);
    return detail::top(pressure(phi, xd, chi0, opt));
}

// P, dP, d2P and, for order 3, d3P at x, starting the gap equations from chi0
template <class Model>
PressureDerivatives derivatives(const Model& phi, const VectorXd& x, const VectorXd& chi0, int order = 2,
                                const NewtonOptions& opt = {}) {
    if (order < 1 || order > 3) throw std::invalid_argument("mfd::derivatives: order must be 1, 2 or 3");
    const Index d = x.size();
    PressureDerivatives r;
    r.chi = gap(phi, x, chi0, opt);
    auto P = [&](const auto& xd) { return pressure(phi, xd, r.chi, opt); };
    if (order == 1) {
        r.P = P(x);
        r.dP = detail::gradient(P, x, 0);
        return r;
    }
    detail::hessian(P, x, 0, r.P, r.dP, r.d2P);
    if (order < 3) return r;
    r.d3P.assign(d, MatrixXd(d, d));
    for (Index a = 0; a < d; ++a)
        for (Index b = a; b < d; ++b)
            for (Index c = b; c < d; ++c) {
                const double v = derivative<3>(phi, x, r.chi, {a, b, c}, opt);
                for (auto [i, j, k] : {std::array<Index, 3>{a, b, c}, {a, c, b}, {b, a, c}, {b, c, a}, {c, a, b},
                                       {c, b, a}})
                    r.d3P[i](j, k) = v;
            }
    return r;
}

}  // namespace mfd
