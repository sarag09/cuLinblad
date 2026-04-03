#include <complex>
#include <stdexcept>
#include <vector>

#include "culindblad/local_operator_utils.hpp"
#include "culindblad/types.hpp"

namespace culindblad {

std::vector<Complex> local_conjugate_transpose(
    const std::vector<Complex>& op,
    Index dim)
{
    if (op.size() != dim * dim) {
        throw std::runtime_error("local_conjugate_transpose: op has wrong size");
    }

    std::vector<Complex> result(dim * dim, Complex{0.0, 0.0});

    for (Index i = 0; i < dim; ++i) {
        for (Index j = 0; j < dim; ++j) {
            result[j * dim + i] = std::conj(op[i * dim + j]);
        }
    }

    return result;
}

std::vector<Complex> local_multiply_square(
    const std::vector<Complex>& A,
    const std::vector<Complex>& B,
    Index dim)
{
    if (A.size() != dim * dim) {
        throw std::runtime_error("local_multiply_square: A has wrong size");
    }

    if (B.size() != dim * dim) {
        throw std::runtime_error("local_multiply_square: B has wrong size");
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

bool try_extract_local_diagonal(
    const std::vector<Complex>& op,
    Index dim,
    std::vector<Complex>& diagonal_out)
{
    if (op.size() != dim * dim) {
        throw std::runtime_error("try_extract_local_diagonal: op has wrong size");
    }

    constexpr double kDiagonalTolerance = 1.0e-14;
    diagonal_out.assign(dim, Complex{0.0, 0.0});

    for (Index i = 0; i < dim; ++i) {
        diagonal_out[i] = op[i * dim + i];

        for (Index j = 0; j < dim; ++j) {
            if (i == j) {
                continue;
            }

            if (std::abs(op[i * dim + j]) > kDiagonalTolerance) {
                diagonal_out.clear();
                return false;
            }
        }
    }

    return true;
}

} // namespace culindblad
