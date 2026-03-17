#include <stdexcept>
#include <vector>

#include "culindblad/cpu_reference.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

std::vector<Complex> multiply_square_matrices(
    ConstStateBuffer A,
    ConstStateBuffer B,
    Index dim)
{
    if (A.size != dim * dim) {
        throw std::runtime_error("multiply_square_matrices: A has wrong size");
    }

    if (B.size != dim * dim) {
        throw std::runtime_error("multiply_square_matrices: B has wrong size");
    }

    std::vector<Complex> C(dim * dim, Complex{0.0, 0.0});

    for (Index i = 0; i < dim; ++i) {
        for (Index k = 0; k < dim; ++k) {
            const Complex a_ik = A[i * dim + k];

            for (Index j = 0; j < dim; ++j) {
                C[i * dim + j] += a_ik * B[k * dim + j];
            }
        }
    }

    return C;
}

} // namespace culindblad