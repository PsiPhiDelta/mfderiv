#pragma once
// Common types, the model interface, and helpers for the nested forward-mode dual numbers of the
// autodiff library (https://autodiff.github.io).
//
// A model is a callable phi(x, chi) that returns the effective potential Phi(x, chi) for the
// external (control) variables x, for example (T, mu), and the internal variables chi, for example
// condensates, gaps and vector mean fields. It must be generic in the scalar type,
//
//     struct Model {
//         template <class S>
//         S operator()(const mfd::VectorX<S>& x, const mfd::VectorX<S>& chi) const;
//     };
//
// so that dual numbers of any nesting depth can pass through it. Elementary functions should be
// called unqualified after `using std::exp;` etc., so that the overloads of autodiff are found by
// argument-dependent lookup. Branches must compare primal values (autodiff's comparison operators
// already do this). Only the scalar types of autodiff are used, not its Eigen helpers, so any Eigen
// version works.

#include <type_traits>
#include <vector>

#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp>

namespace mfd {

template <class S>
using VectorX = Eigen::Matrix<S, Eigen::Dynamic, 1>;
template <class S>
using MatrixX = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Pressure and its derivatives with respect to the control variables at one point
struct PressureDerivatives {
    double P = 0.0;
    VectorXd dP;                 // dP/dx_a, the densities
    MatrixXd d2P;                // d2P/dx_a dx_b, the susceptibilities
    std::vector<MatrixXd> d3P;   // d3P[a](b, c) = d3P/dx_a dx_b dx_c (order 3 only)
    VectorXd chi;                // chibar, the solution of the gap equations
    MatrixXd S;                  // dchibar/dx (closed forms only)
    MatrixXd d2P_naive;          // -Phi_xx at fixed chi, without the response of chi (closed forms only)
};

namespace ad {

// One layer of a nested dual number S = (value, tangent), both of the inner type.
template <class S>
struct layer {
    static constexpr bool is_dual = false;
    using inner = S;
};
template <class T>
struct layer<autodiff::detail::Dual<T, T>> {
    static constexpr bool is_dual = true;
    using inner = T;
};
template <class S>
inline constexpr bool is_dual_v = layer<S>::is_dual;
template <class S>
using inner_t = typename layer<S>::inner;

// number of dual layers, 0 for double
template <class S>
constexpr int depth() {
    if constexpr (is_dual_v<S>)
        return 1 + depth<inner_t<S>>();
    else
        return 0;
}

template <class S>
S make(const inner_t<S>& value, const inner_t<S>& tangent) {
    S r;
    r.val = value;
    r.grad = tangent;
    return r;
}

// embed a value of a less nested type (for example double) with zero tangents on every outer layer
template <class To, class From>
To lift(const From& v) {
    if constexpr (std::is_same_v<To, From>)
        return v;
    else
        return make<To>(lift<inner_t<To>>(v), inner_t<To>{});
}

// innermost primal value
template <class S>
double primal(const S& v) {
    if constexpr (is_dual_v<S>)
        return primal(v.val);
    else
        return v;
}

template <class S>
VectorX<inner_t<S>> value(const VectorX<S>& v) {
    VectorX<inner_t<S>> r(v.size());
    for (Index i = 0; i < v.size(); ++i) r(i) = v(i).val;
    return r;
}
template <class S>
VectorX<inner_t<S>> tangent(const VectorX<S>& v) {
    VectorX<inner_t<S>> r(v.size());
    for (Index i = 0; i < v.size(); ++i) r(i) = v(i).grad;
    return r;
}
template <class S>
MatrixX<inner_t<S>> value(const MatrixX<S>& m) {
    MatrixX<inner_t<S>> r(m.rows(), m.cols());
    for (Index i = 0; i < m.rows(); ++i)
        for (Index j = 0; j < m.cols(); ++j) r(i, j) = m(i, j).val;
    return r;
}
template <class S>
MatrixX<inner_t<S>> tangent(const MatrixX<S>& m) {
    MatrixX<inner_t<S>> r(m.rows(), m.cols());
    for (Index i = 0; i < m.rows(); ++i)
        for (Index j = 0; j < m.cols(); ++j) r(i, j) = m(i, j).grad;
    return r;
}
template <class S>
VectorX<S> make(const VectorX<inner_t<S>>& value, const VectorX<inner_t<S>>& tangent) {
    VectorX<S> r(value.size());
    for (Index i = 0; i < value.size(); ++i) r(i) = make<S>(value(i), tangent(i));
    return r;
}
template <class To, class From>
VectorX<To> lift(const VectorX<From>& v) {
    VectorX<To> r(v.size());
    for (Index i = 0; i < v.size(); ++i) r(i) = lift<To>(v(i));
    return r;
}
template <class S>
VectorXd primal(const VectorX<S>& v) {
    VectorXd r(v.size());
    for (Index i = 0; i < v.size(); ++i) r(i) = primal(v(i));
    return r;
}

}  // namespace ad

namespace detail {

// Value, gradient and Hessian of f with respect to the variables marked by `active`, for inputs of
// any scalar type I (double or a dual number). The derivatives are taken with one more nested layer
// on top of I, so the results carry all tangents of I. f is called as f(v) with v of type
// VectorX<Dual<Dual<I, I>, Dual<I, I>>>, and the active variables are those with index >= first.
template <class F, class I>
void hessian(const F& f, const VectorX<I>& v, Index first, I& value, VectorX<I>& grad, MatrixX<I>& hess) {
    using D1 = autodiff::detail::Dual<I, I>;
    using D2 = autodiff::detail::Dual<D1, D1>;
    const Index n = v.size(), m = n - first;
    VectorX<D2> w = ad::lift<D2>(v);
    grad.resize(m);
    hess.resize(m, m);
    const I one = ad::lift<I>(1.0), zero{};
    for (Index i = 0; i < m; ++i) {
        w(first + i).val.grad = one;
        for (Index j = 0; j <= i; ++j) {
            w(first + j).grad.val = one;
            const D2 r = f(w);
            w(first + j).grad.val = zero;
            hess(i, j) = r.grad.grad;
            hess(j, i) = r.grad.grad;
            if (j == 0) {
                grad(i) = r.val.grad;
                value = r.val.val;
            }
        }
        w(first + i).val.grad = zero;
    }
    if (m == 0) value = f(w).val.val;
}

// gradient of f with respect to the variables with index >= first, for inputs of any scalar type S
template <class F, class S>
VectorX<S> gradient(const F& f, const VectorX<S>& v, Index first) {
    using D1 = autodiff::detail::Dual<S, S>;
    const Index n = v.size(), m = n - first;
    VectorX<D1> w = ad::lift<D1>(v);
    VectorX<S> grad(m);
    const S one = ad::lift<S>(1.0), zero{};
    for (Index i = 0; i < m; ++i) {
        w(first + i).grad = one;
        grad(i) = f(w).grad;
        w(first + i).grad = zero;
    }
    return grad;
}

// Solve (sum_k eps_k H_k) y = b for nested dual matrices, with the LU factors of the primal matrix.
// Each layer needs two solves with the same factors, (H0 + eps H1)(y0 + eps y1) = b0 + eps b1.
template <class I>
VectorX<I> solve(const Eigen::PartialPivLU<MatrixXd>& lu, const MatrixX<I>& H, const VectorX<I>& b) {
    if constexpr (!ad::is_dual_v<I>) {
        return lu.solve(b);
    } else {
        using J = ad::inner_t<I>;
        const MatrixX<J> H0 = ad::value(H), H1 = ad::tangent(H);
        const VectorX<J> y0 = solve(lu, H0, ad::value(b));
        VectorX<J> rhs = ad::tangent(b);
        for (Index i = 0; i < rhs.size(); ++i)
            for (Index k = 0; k < rhs.size(); ++k) rhs(i) -= H1(i, k) * y0(k);
        return ad::make<I>(y0, solve(lu, H0, rhs));
    }
}

template <class S>
MatrixXd primal(const MatrixX<S>& m) {
    MatrixXd r(m.rows(), m.cols());
    for (Index i = 0; i < m.rows(); ++i)
        for (Index j = 0; j < m.cols(); ++j) r(i, j) = ad::primal(m(i, j));
    return r;
}

// f(v) = phi(v.head(d), v.tail(N))
template <class Model>
struct Joined {
    const Model& phi;
    Index d;
    template <class Vec>
    auto operator()(const Vec& v) const {
        using S = typename Vec::Scalar;
        const VectorX<S> x = v.head(d);
        const VectorX<S> chi = v.tail(v.size() - d);
        return phi(x, chi);
    }
};

template <class S>
VectorX<S> concat(const VectorX<S>& a, const VectorX<S>& b) {
    VectorX<S> r(a.size() + b.size());
    for (Index i = 0; i < a.size(); ++i) r(i) = a(i);
    for (Index i = 0; i < b.size(); ++i) r(a.size() + i) = b(i);
    return r;
}

}  // namespace detail
}  // namespace mfd
