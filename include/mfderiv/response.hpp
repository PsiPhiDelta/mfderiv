#pragma once
// Response functions from the derivatives of the pressure at fixed control variables. The control
// variables are x = (T, mu_1, ..., mu_k), or x = (mu_1, ..., mu_k) at T = 0, with the densities
// sigma = dP/dx = (s, n_1, ..., n_k) and the susceptibilities chi = d2P/dx2. Each response is a
// derivative along a thermodynamic process, d A/d B at fixed C_1, ..., C_{d-1}, which is a ratio of
// Jacobian determinants in the space of control variables.

#include <stdexcept>
#include <vector>

#include "core.hpp"

namespace mfd {

inline double enthalpy_density(const VectorXd& x, const VectorXd& sigma) { return x.dot(sigma); }

inline double energy_density(const VectorXd& x, double P, const VectorXd& sigma) { return -P + x.dot(sigma); }

// c_s^2 = dP/d epsilon at fixed s/n_i = sigma^T chi^{-1} sigma / (x . sigma). At T = 0 with x = (mu)
// this is n / (mu dn/dmu).
inline double sound_speed2(const VectorXd& x, const VectorXd& sigma, const MatrixXd& chi) {
    const VectorXd u = chi.fullPivLu().solve(sigma);  // tangent of the adiabat
    return sigma.dot(u) / x.dot(sigma);
}

// Gamma = d ln P / d ln n at fixed s/n_i
inline double adiabatic_index(const VectorXd& x, double P, const VectorXd& sigma, const MatrixXd& chi) {
    return sound_speed2(x, sigma, chi) * x.dot(sigma) / P;
}

// c_V = T ds/dT at fixed n_i = T / (chi^{-1})_TT, for x = (T, mu_1, ...)
inline double heat_capacity_V(const VectorXd& x, const MatrixXd& chi) {
    const VectorXd e = chi.fullPivLu().solve(VectorXd::Unit(x.size(), 0));
    return x(0) / e(0);
}

// (dP/d epsilon at fixed n_i, dP/dn_i at fixed epsilon and n_{j != i}) = sigma^T chi^{-1} A^{-1},
// with A = d(epsilon, n)/d(T, mu) chi^{-1} = [[T, mu^T], [0, 1]], for x = (T, mu_1, ...)
inline VectorXd hydro_derivatives(const VectorXd& x, const VectorXd& sigma, const MatrixXd& chi) {
    const Index d = x.size();
    MatrixXd A = MatrixXd::Identity(d, d);
    A.row(0) = x.transpose();
    const VectorXd w = chi.transpose().fullPivLu().solve(sigma);
    return A.transpose().fullPivLu().solve(w);
}

// Susceptibilities after eliminating the control variables in `fixed`, whose densities are held at
// zero (for example strangeness or charge neutrality), as the Schur complement of their block.
inline MatrixXd constrain(const MatrixXd& chi, const std::vector<Index>& fixed) {
    std::vector<Index> keep;
    for (Index i = 0; i < chi.rows(); ++i) {
        bool f = false;
        for (Index j : fixed) f = f || (i == j);
        if (!f) keep.push_back(i);
    }
    const Index k = keep.size(), m = fixed.size();
    MatrixXd A(k, k), B(k, m), C(m, m);
    for (Index i = 0; i < k; ++i) {
        for (Index j = 0; j < k; ++j) A(i, j) = chi(keep[i], keep[j]);
        for (Index j = 0; j < m; ++j) B(i, j) = chi(keep[i], fixed[j]);
    }
    for (Index i = 0; i < m; ++i)
        for (Index j = 0; j < m; ++j) C(i, j) = chi(fixed[i], fixed[j]);
    if (m == 0) return A;
    return A - B * C.fullPivLu().solve(B.transpose());
}

// dA/dB along the process that keeps C_1, ..., C_{d-1} fixed, from the gradients of A, B and the
// C_i with respect to the control variables (columns of gradC)
inline double process_derivative(const VectorXd& gradA, const VectorXd& gradB, const MatrixXd& gradC) {
    const Index d = gradA.size();
    if (gradC.cols() != d - 1) throw std::invalid_argument("mfd::process_derivative: need d - 1 constraints");
    MatrixXd JA(d, d), JB(d, d);
    JA.col(0) = gradA;
    JB.col(0) = gradB;
    JA.rightCols(d - 1) = gradC;
    JB.rightCols(d - 1) = gradC;
    return JA.determinant() / JB.determinant();
}

}  // namespace mfd
