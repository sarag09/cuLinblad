#include "culindblad/pulse_envelope.hpp"

#include <cmath>
#include <stdexcept>

namespace culindblad {

namespace {

// Same literal as Julia's Float64 pi, so the sin/2tr arguments match bit-for-bit.
constexpr double kPi = 3.141592653589793238462643383279502884;

// Mirrors gate_sim.jl _interp_linear (searchsortedfirst + endpoint clamp). x is monotone ascending.
double interp_linear(const std::vector<double>& x, const std::vector<double>& y, double xq) {
    const std::size_t n = x.size();
    if (xq <= x[0]) return y[0];
    if (xq >= x[n - 1]) return y[n - 1];
    std::size_t i = 1;
    while (i < n && x[i] < xq) ++i;  // first i with x[i] >= xq  => x[i-1] < xq <= x[i]
    const double w = (xq - x[i - 1]) / (x[i] - x[i - 1]);
    return y[i - 1] + w * (y[i] - y[i - 1]);
}

// gate_sim.jl _slepian_omega_c
double slepian_omega_c(double theta, double w1, double K, double cg) {
    const double tt = std::tan(theta);
    const double t2 = tt * tt;
    const double a = cg * cg * K * K;
    return (w1 / (2.0 * t2)) * (2.0 * t2 + a + cg * K * std::sqrt(a + 4.0 * t2));
}

} // namespace

TimeScalarFunction make_envelope(const EnvelopeSpec& s) {
    if (s.shape == "sine") {
        const double a = s.a, b = s.b, T = s.T;
        return [a, b, T](double t) {
            return (0.0 < t && t < T) ? (a - b) * std::sin(kPi * t / T) + b : b;
        };
    }
    if (s.shape == "flattop") {
        const double a = s.a, b = s.b, T = s.T, tr = s.tr;
        return [a, b, T, tr](double t) {
            if (t <= 0.0 || t >= T) return b;
            double sv;
            if (t < tr) {
                const double q = std::sin(kPi * t / (2.0 * tr));
                sv = q * q;
            } else if (t > T - tr) {
                const double q = std::sin(kPi * (T - t) / (2.0 * tr));
                sv = q * q;
            } else {
                sv = 1.0;
            }
            return b + (a - b) * sv;
        };
    }
    if (s.shape == "flat_top_gaussian") {
        // a is the pre-normalized a_eff (the Julia constructor already applied the plateau renorm).
        const double a = s.a, b = s.b, t0 = s.t0, tau = s.tau, sigma = s.sigma;
        return [a, b, t0, tau, sigma](double t) {
            const double sc = 1.0 / (std::sqrt(2.0) * sigma);
            const double env = 0.5 * (std::erf((t0 + tau - t) * sc) - std::erf((t0 - t) * sc));
            return b + (a - b) * env;
        };
    }
    if (s.shape == "slepian") {
        const double w1 = s.omega1, K = s.K, cg = s.cg, T = s.T, idle = s.omega_c_idle;
        const std::vector<double> u = s.ugrid;
        const std::vector<double> th = s.thetagrid;
        if (u.size() < 2 || u.size() != th.size()) {
            throw std::runtime_error("make_envelope: slepian needs matching ugrid/thetagrid (>=2 pts)");
        }
        return [w1, K, cg, T, idle, u, th](double t) {
            if (t <= 0.0 || t >= T) return idle;
            const double theta = interp_linear(u, th, t / T);
            return slepian_omega_c(theta, w1, K, cg);
        };
    }
    throw std::runtime_error("make_envelope: unknown shape '" + s.shape + "'");
}

} // namespace culindblad
