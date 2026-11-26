# mfderiv

Derivatives of the pressure for thermal field theories whose grand potential is the value of an
effective potential at a stationary point, such as mean-field models (NJL, quark-meson, chiral
mean-field), 2PI and other variational truncations. Densities, susceptibilities, the speed of sound
and higher derivatives are computed with forward-mode automatic differentiation, to machine
precision and without finite differences of a tabulated pressure.

A model is a single function, the effective potential `Phi(x, chi)` of the control variables `x`
(for example `T` and `mu`) and the internal variables `chi` (condensates, gaps, vector mean fields,
chemical potentials fixed by neutrality). The pressure is `P(x) = -Phi(x, chibar(x))`, where
`chibar(x)` solves the gap equations `dPhi/dchi = 0`. Because `chibar(x)` is known only implicitly,
automatic differentiation of a pressure routine either misses the response of `chi` or has to
differentiate through every iteration of the solver. This library uses the implicit-function theorem
instead.

- **Gap solver aware of the implicit-function theorem.** `mfd::gap` and `mfd::pressure` accept dual
  numbers of any nesting depth. Each layer costs one Newton step on top of the converged solution, so
  `P(x)` becomes an ordinary differentiable function, and derivatives of any order follow.
- **Closed formulas.** Envelope theorem, Schur complement and tangent contraction give the
  derivatives up to third order from partial derivatives of `Phi` at one point.
- **Response functions.** Speed of sound at fixed entropy per particle, heat capacity, adiabatic
  index, derivatives at fixed density or energy density, and Schur complements for neutrality
  conditions.

## History

The method grew out of symbolic pressure expansion coefficients for mean-field models,
[PressureExpansion](https://github.com/PsiPhiDelta/PressureExpansion) (Mathematica, 2025), and is
described in H. Gholami, *Higher-order pressure derivatives in thermal field theories*, with a Julia
reference implementation. This library is its C++ implementation, intended for C++ equation-of-state
codes, for example modules of the MUSES framework.

## Requirements

- C++17
- [Eigen](https://eigen.tuxfamily.org) 3.3 or later (Eigen 5 works)
- [autodiff](https://autodiff.github.io) headers. Only its scalar types are used, not its Eigen
  helpers. CMake downloads them when they are not found (`MFDERIV_FETCH_AUTODIFF=ON`, the default).

## Usage

Write the potential once, generic in the scalar type. Call elementary functions unqualified after
`using std::exp;` and the like, so that the overloads for dual numbers are found.

```cpp
#include <mfderiv/mfderiv.hpp>

// massless quarks with a repulsive vector interaction, chi(0) = mu - 2 G_V n
struct VectorGas {
    double GV = 5e-6;  // MeV^-2
    template <class S>
    S operator()(const mfd::VectorX<S>& x, const mfd::VectorX<S>& chi) const {
        const double pi = 3.14159265358979323846;
        const S& T = x(0);
        const S& mu = x(1);
        const S& m = chi(0);
        const S P = 12.0 * (7.0 * pi * pi * T * T * T * T / 360.0 + m * m * T * T / 12.0
                            + m * m * m * m / (24.0 * pi * pi));
        return -P - (mu - m) * (mu - m) / (4.0 * GV);
    }
};

VectorGas phi;
mfd::VectorXd x(2), chi0(1);
x << 40.0, 400.0;   // T, mu in MeV
chi0 << 350.0;      // starting point of the gap equations

auto r = mfd::derivatives(phi, x, chi0, 3);   // P, dP = (s, n), d2P, d3P and chibar
double cs2 = mfd::sound_speed2(x, r.dP, r.d2P);
double c04 = mfd::derivative<4>(phi, x, r.chi, {1, 1, 1, 1});   // d^4 P / d mu^4
```

At `T = 0` use `x = (mu)` (or the `mu` block of the results), where `c_s^2 = n / (mu dn/dmu)`.

| function | content |
|---|---|
| `newton(phi, x, chi0)` | Newton iteration for the gap equations at fixed `x` (double) |
| `gap(phi, x, chi0)` | `chibar(x)` for `x` of type double or any nested `autodiff` dual type |
| `pressure(phi, x, chi0)` | `P(x) = -Phi(x, chibar(x))`, differentiable in `x` |
| `derivatives(phi, x, chi0, order)` | `P`, `dP`, `d2P`, `d3P` through the gap solver |
| `derivative<K>(phi, x, chi0, idx)` | a single mixed derivative of order `K` |
| `closed_form_derivatives(phi, x, chibar, order)` | the same up to third order from closed formulas, with `S = dchibar/dx` and the naive `-Phi_xx` |
| `sound_speed2`, `adiabatic_index`, `heat_capacity_V`, `hydro_derivatives` | response functions from `dP` and `d2P` |
| `constrain(chi, fixed)` | susceptibilities after fixing densities to zero (neutrality) |
| `process_derivative(gradA, gradB, gradC)` | `dA/dB` along any process that keeps `C_1, ..., C_{d-1}` fixed |

## Notes

- The gap equations are stationarity conditions. Variables that maximize `Phi`, such as repulsive
  vector fields and chemical potentials fixed by neutrality, are handled like the condensates.
  Branch selection among several solutions is up to the caller, through `chi0`.
- Branches inside `Phi` must compare primal values. The comparison operators of `autodiff` do this.
- Momentum integrals with Fermi surfaces should be split at the Fermi momenta, which move with `x` and
  `chi`. The quadrature is then smooth in these variables, and the derivatives of the discretized
  integral include the boundary terms.
- The internal Hessian must be invertible. It is singular at spinodals, critical points and
  second-order transitions, where the derivatives of the pressure diverge or jump.

## Building the tests

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

The tests compare with a model whose derivatives are known in closed form (up to fourth order),
with finite differences, and the two routes with each other.

## License

MIT, see [LICENSE](LICENSE). If you use the library, please cite it, see [CITATION.cff](CITATION.cff).
