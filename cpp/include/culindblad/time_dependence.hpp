#pragma once

#include <functional>

namespace culindblad {

using TimeScalarFunction = std::function<double(double)>;

double evaluate_time_scalar(
    const TimeScalarFunction& f,
    double t);

TimeScalarFunction make_cosine_time_scalar(
    double amplitude,
    double omega,
    double phase);

} // namespace culindblad
