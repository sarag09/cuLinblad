#pragma once

#include <string>
#include <vector>

#include "culindblad/time_dependence.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

struct TimeDependentTerm {
    std::string name;
    std::vector<Index> sites;
    std::vector<Complex> matrix;
    Index rows;
    Index cols;
    TimeScalarFunction coefficient;
};

double evaluate_time_dependent_coefficient(
    const TimeDependentTerm& td_term,
    double t);

} // namespace culindblad