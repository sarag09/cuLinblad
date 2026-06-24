#pragma once

#include <string>
#include <vector>

#include "culindblad/time_dependence.hpp"

namespace culindblad {

// Parsed envelope spec: shape id + the BUILT-struct numeric fields. The Julia exporter serializes the
// already-constructed gate_sim.jl envelope's fields (e.g. FlatTopGaussianEnv's pre-normalized a_eff,
// SlepianEnv's u->theta grid), so C++ replays the raw formula and never re-runs the Julia constructors.
struct EnvelopeSpec {
    std::string shape;            // "sine" | "flattop" | "flat_top_gaussian" | "slepian"
    double a = 0.0, b = 0.0, T = 0.0;     // sine / flattop (a = hold, b = idle)
    double tr = 0.0;                       // flattop ramp duration
    double t0 = 0.0, tau = 0.0, sigma = 0.0;  // flat_top_gaussian (a is a_eff)
    double omega1 = 0.0, K = 0.0, cg = 0.0, omega_c_idle = 0.0;  // slepian inversion params
    std::vector<double> ugrid;             // slepian: monotone u = t/T in [0,1]
    std::vector<double> thetagrid;         // slepian: theta at each u
};

// Build the TimeScalarFunction for an envelope, exactly mirroring gate_sim.jl
// (SineEnv / FlatTopEnv / FlatTopGaussianEnv / SlepianEnv). Throws on an unknown shape or bad grid.
TimeScalarFunction make_envelope(const EnvelopeSpec& spec);

} // namespace culindblad
