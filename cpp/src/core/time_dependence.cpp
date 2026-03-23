#include "culindblad/time_dependence.hpp"

#include <cmath>
#include <functional>

namespace culindblad {

double evaluate_time_scalar(
    const TimeScalarFunction& f,
    double t)
{
    return f(t);
}

TimeScalarFunction make_cosine_time_scalar(
    double amplitude,
    double omega,
    double phase)
{
    return [amplitude, omega, phase](double t) {
        return amplitude * std::cos(omega * t + phase);
    };
}

} // namespace culindblad
