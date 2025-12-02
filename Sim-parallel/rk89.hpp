// rk89.hpp
#pragma once

#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstddef>

namespace rk89 {

// State type
using State = std::vector<double>;

// Right-hand side: dydt = f(t, y)
using RHS = std::function<void(double t, const State& y, State& dydt)>;

// Generic embedded Runge–Kutta tableau
struct RKTableau {
    int s;                            // number of stages
    std::vector<double> c;            // size s
    std::vector<double> a;            // size s*s, row-major, only j < i used
    std::vector<double> b_high;       // size s (higher order, e.g. 9)
    std::vector<double> b_low;        // size s (lower order, e.g. 8)
    int order_high;                   // e.g. 9
    int order_low;                    // e.g. 8
};

// Helper for accessing a(i,j) in flat row-major storage
inline double a_ij(const RKTableau& tab, int i, int j) {
    return tab.a[static_cast<std::size_t>(i) * tab.s + j];
}

// Clamp helper
inline double clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/**
 * Perform one embedded RK step using an arbitrary tableau.
 *
 * Inputs:
 *   f      : RHS functor, dydt = f(t, y)
 *   tab    : RK tableau
 *   t      : current time
 *   h      : step size
 *   y      : current state
 *
 * Outputs:
 *   y_high : high-order solution y^{(p)}(t + h)
 *   y_err  : estimate y^{(p)} - y^{(p-1)} (componentwise)
 */
inline void rk_embedded_step(
    const RHS& f,
    const RKTableau& tab,
    double t,
    double h,
    const State& y,
    State& y_high,
    State& y_err
) {
    const int s = tab.s;
    const std::size_t n = y.size();

    if (tab.c.size() != static_cast<std::size_t>(s) ||
        tab.b_high.size() != static_cast<std::size_t>(s) ||
        tab.b_low.size()  != static_cast<std::size_t>(s) ||
        tab.a.size()      != static_cast<std::size_t>(s * s)) {
        throw std::runtime_error("RKTableau has inconsistent sizes");
    }

    // Stage derivatives k[i][n]
    std::vector<State> k(s, State(n, 0.0));
    State ytmp(n);

    // Compute stages
    for (int i = 0; i < s; ++i) {
        // ytmp = y + h * sum_{j=0}^{i-1} a[i,j] * k[j]
        for (std::size_t m = 0; m < n; ++m) {
            double sum = 0.0;
            for (int j = 0; j < i; ++j) {
                sum += a_ij(tab, i, j) * k[j][m];
            }
            ytmp[m] = y[m] + h * sum;
        }

        const double ti = t + tab.c[static_cast<std::size_t>(i)] * h;
        f(ti, ytmp, k[i]);  // k[i] = f(ti, ytmp)
    }

    // Combine stages into high-order solution and error estimate
    y_high.assign(n, 0.0);
    y_err.assign(n, 0.0);

    for (std::size_t m = 0; m < n; ++m) {
        double sum_high = 0.0;
        double sum_low  = 0.0;
        for (int i = 0; i < s; ++i) {
            sum_high += tab.b_high[static_cast<std::size_t>(i)] * k[i][m];
            sum_low  += tab.b_low [static_cast<std::size_t>(i)] * k[i][m];
        }
        const double yh = y[m] + h * sum_high;
        const double yl = y[m] + h * sum_low;
        y_high[m] = yh;
        y_err[m]  = yh - yl;  // high - low
    }
}

/**
 * Adaptive integration using an embedded RK p/(p-1) pair (e.g. 9/8).
 *
 * Returns:
 *   0  on success
 *   2  if too many steps
 *   3  if step size underflow
 */
inline int integrate_adaptive(
    const RHS& f,
    const RKTableau& tab,
    double t0,
    double t1,
    State& y,                // in: y(t0); out: y(t1)
    double h_init,           // initial step size guess (sign can be wrong)
    double atol,             // absolute tolerance
    double rtol,             // relative tolerance
    double h_min,            // minimum |h|
    double h_max,            // maximum |h|
    int    max_steps         // maximum accepted steps
) {
    if (y.empty()) {
        throw std::runtime_error("State vector y is empty");
    }

    const double direction = (t1 >= t0) ? 1.0 : -1.0;
    double h = h_init;

    // Ensure h has correct sign and is non-zero
    if (h * direction <= 0.0) {
        h = direction * std::abs((t1 - t0) * 0.1);  // heuristic
    }

    h_min = std::abs(h_min);
    h_max = std::abs(h_max);

    State y_new(y.size());
    State y_err(y.size());

    const double safety     = 0.9;
    const double min_factor = 0.2;
    const double max_factor = 5.0;
    const double order      = static_cast<double>(tab.order_high);
    const double inv_order  = 1.0 / order;

    double t = t0;
    int steps = 0;

    while (true) {
        if (steps > max_steps) {
            return 2; // too many steps
        }

        // Prevent overshoot
        if (direction > 0.0) {
            if (t + h > t1) h = t1 - t;
        } else {
            if (t + h < t1) h = t1 - t;
        }

        if (std::abs(h) < h_min) {
            return 3; // step size underflow
        }
        if (std::abs(h) > h_max) {
            h = direction * h_max;
        }

        // Trial step
        rk_embedded_step(f, tab, t, h, y, y_new, y_err);

        // RMS error norm
        double err_norm = 0.0;
        const std::size_t n = y.size();
        for (std::size_t i = 0; i < n; ++i) {
            double sc = atol + rtol * std::max(std::abs(y[i]), std::abs(y_new[i]));
            double ei = y_err[i] / sc;
            err_norm += ei * ei;
        }
        err_norm = std::sqrt(err_norm / static_cast<double>(n));

        if (err_norm <= 1.0) {
            // Accept
            y = y_new;
            t += h;
            ++steps;

            // Done?
            if ((direction > 0.0 && t >= t1) ||
                (direction < 0.0 && t <= t1)) {
                break;
            }

            // Propose new h
            if (err_norm == 0.0) {
                h *= max_factor;
            } else {
                double factor = safety * std::pow(1.0 / err_norm, inv_order);
                factor = clamp(factor, min_factor, max_factor);
                h *= factor;
            }
        } else {
            // Reject, shrink h
            double factor = safety * std::pow(1.0 / err_norm, inv_order);
            factor = clamp(factor, min_factor, max_factor);
            h *= factor;
        }
    }

    return 0; // success
}

/**
 * Factory for a "default" RK 8/9 (actually 9/8) tableau.
 *
 * ⚠️ IMPORTANT:
 *   This function is structurally complete but the numerical coefficients
 *   are left as TODOs. You must paste a real 8/9 or 9/8 Butcher tableau
 *   (c, a, b_high, b_low) that you are allowed to use.
 *
 * E.g., one source is - https://www.sfu.ca/~jverner/
 *
 */
inline RKTableau make_default_rk89()
{
    // Many 9(8) pairs use 16 stages; adjust if your chosen method is different.
    constexpr int S = 16;

    RKTableau tab;
    tab.s = S;
    tab.c.resize(S);
    tab.a.assign(static_cast<std::size_t>(S) * S, 0.0);  // initialize all zeros
    tab.b_high.resize(S);
    tab.b_low.resize(S);
    tab.order_high = 9;
    tab.order_low  = 8;

    // =========================  IMPORTANT  =========================
    // Paste your actual coefficients here from a published 8/9 or 9/8 scheme.
    //
    // Example mapping (assuming 1-based indices in the paper/file):
    //
    //   c_i (nodes):
    //       tab.c[0] = c1;
    //       tab.c[1] = c2;
    //       ...
    //       tab.c[S-1] = cS;
    //
    //   a_{i,j} (Butcher matrix, j < i):
    //       tab.a[1*S + 0] = a_2,1;
    //       tab.a[2*S + 0] = a_3,1;
    //       tab.a[2*S + 1] = a_3,2;
    //       ...
    //
    //   b_high[i] (high-order weights, order 9):
    //       tab.b_high[0] = b1^(9);
    //       ...
    //       tab.b_high[S-1] = bS^(9);
    //
    //   b_low[i] (low-order weights, order 8):
    //       tab.b_low[0] = b1^(8);
    //       ...
    //       tab.b_low[S-1] = bS^(8);
    //
    // ===============================================================

    // ----- PLACEHOLDER CONTENT (compiles but NOT a real 8/9 method) -----
    // This is ONLY so the code builds; replace every line in this block
    // with real coefficients for actual use.

    for (int i = 0; i < S; ++i) {
        tab.c[static_cast<std::size_t>(i)]      = 0.0;
        tab.b_high[static_cast<std::size_t>(i)] = 0.0;
        tab.b_low[static_cast<std::size_t>(i)]  = 0.0;
    }

    // Trivial "do nothing" weights; again, you MUST replace these.
    tab.c[0]      = 0.0;
    tab.b_high[0] = 1.0;
    tab.b_low[0]  = 1.0;

    // -----------------------------------------------------------------

    return tab;
}

} // namespace rk89

