// main.cpp
#include <iostream>
#include <iomanip>
#include <cmath>

#include "rk87.hpp"

int main() {
    using namespace rk87;

    // ODE: y' = sin(t), y(0) = 0  =>  y(t) = ∫_0^t sin(x) dx
    RHS f = [](double t, const State& y, State& dydt) {
        (void)y;                // y not used in this ODE
        dydt.resize(1);
        dydt[0] = std::sin(t);
    };

    State y(1);
    y[0] = 0.0;

    const double t0 = 0.0;
    const double t1 = M_PI;     // integrate from 0 to π

    RKTableau tab = make_default_rk87();  // TODO: fill real coefficients

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

