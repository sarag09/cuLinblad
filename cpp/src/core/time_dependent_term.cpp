#include "culindblad/time_dependent_term.hpp"

#include "culindblad/time_dependence.hpp"

namespace culindblad {

double evaluate_time_dependent_coefficient(
    const TimeDependentTerm& td_term,
    double t)
{
    return evaluate_time_scalar(td_term.coefficient, t);
}

} // namespace culindblad