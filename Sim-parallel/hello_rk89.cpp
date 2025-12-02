// main.cpp
#include <iostream>
#include <iomanip>
#include <cmath>

#include "rk89.hpp"

int main() {
    using namespace rk89;

    // We want to compute I = ∫_0^π sin(x) dx
    // Model this as y' = sin(t), y(0) = 0, so y(π) = I.

    RHS f = [](double t, const State& y, State& dydt) {
        (void)y;                // y is not used in this ODE
        dydt.resize(1);
        dydt[0] = std::sin(t);  // y' = sin(t)
    };

    State y(1);
    y[0] = 0.0;                 // integral starts at 0

    const double t0 = 0.0;
    const double t1 = M_PI;     // integrate from 0 to π

    // Build the RK 8/9 tableau (you must fill real coefficients in rk89.hpp)
    RKTableau tab = make_default_rk89();

    int status = integrate_adaptive(
        f,
        tab,
        t0,
        t1,
        y,
        /* h_init    */ 0.01,
        /* atol      */ 1e-10,
        /* rtol      */ 1e-10,
        /* h_min     */ 1e-12,
        /* h_max     */ 0.1,
        /* max_steps */ 100000
    );

    if (status != 0) {
        std::cerr << "Integration failed, status = " << status << "\n";
        return 1;
    }

    const double numeric = y[0];
    const double exact   = 2.0;   // ∫_0^π sin(x) dx = 2

    std::cout << std::setprecision(15);
    std::cout << "Integral ∫_0^π sin(x) dx ≈ " << numeric << "\n";
    std::cout << "Exact value               = " << exact   << "\n";
    std::cout << "Absolute error            = " << std::fabs(numeric - exact) << "\n";

    return 0;
}

