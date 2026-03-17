#include <stdexcept>
#include <vector>
#include <complex>

#include "culindblad/cpu_reference.hpp"
#include "culindblad/liouvillian_terms.hpp"

namespace culindblad {

std::vector<Complex> conjugate_transpose(
    const std::vector<Complex>& A,
    Index dim)
{
    if (A.size() != dim * dim) {
        throw std::runtime_error("conjugate_transpose: A has wrong size");
    }

    std::vector<Complex> result(dim * dim, Complex{0.0, 0.0});

    for (Index i = 0; i < dim; ++i) {
        for (Index j = 0; j < dim; ++j) {
            result[j * dim + i] = std::conj(A[i * dim + j]);
        }
    }

    return result;
}    

std::vector<Complex> apply_hamiltonian_commutator(
    const std::vector<Complex>& H,
    ConstStateBuffer rho,
    Index dim)
{
    if (H.size() != dim * dim) {
        throw std::runtime_error("apply_hamiltonian_commutator: H has wrong size");
    }

    if (rho.size != dim * dim) {
        throw std::runtime_error("apply_hamiltonian_commutator: rho has wrong size");
    }

    ConstStateBuffer H_buf{H.data(), H.size()};
    ConstStateBuffer rho_buf{rho.data, rho.size};

    const std::vector<Complex> H_rho = multiply_square_matrices(H_buf, rho_buf, dim);
    const std::vector<Complex> rho_H = multiply_square_matrices(rho_buf, H_buf, dim);

    std::vector<Complex> result(dim * dim, Complex{0.0, 0.0});
    const Complex minus_i{0.0, -1.0};

    for (Index idx = 0; idx < dim * dim; ++idx) {
        result[idx] = minus_i * (H_rho[idx] - rho_H[idx]);
    }

    return result;
}

std::vector<Complex> apply_dissipator(
    const std::vector<Complex>& L,
    ConstStateBuffer rho,
    Index dim)
{
    if (L.size() != dim * dim) {
        throw std::runtime_error("apply_dissipator: L has wrong size");
    }

    if (rho.size != dim * dim) {
        throw std::runtime_error("apply_dissipator: rho has wrong size");
    }

    ConstStateBuffer L_buf{L.data(), L.size()};
    ConstStateBuffer rho_buf{rho.data, rho.size};

    const std::vector<Complex> L_dag = conjugate_transpose(L, dim);
    ConstStateBuffer L_dag_buf{L_dag.data(), L_dag.size()};

    const std::vector<Complex> L_rho = multiply_square_matrices(L_buf, rho_buf, dim);
    ConstStateBuffer L_rho_buf{L_rho.data(), L_rho.size()};

    const std::vector<Complex> jump = multiply_square_matrices(L_rho_buf, L_dag_buf, dim);

    const std::vector<Complex> L_dag_L = multiply_square_matrices(L_dag_buf, L_buf, dim);
    ConstStateBuffer L_dag_L_buf{L_dag_L.data(), L_dag_L.size()};

    const std::vector<Complex> left = multiply_square_matrices(L_dag_L_buf, rho_buf, dim);
    const std::vector<Complex> right = multiply_square_matrices(rho_buf, L_dag_L_buf, dim);

    std::vector<Complex> result(dim * dim, Complex{0.0, 0.0});
    const Complex half{0.5, 0.0};

    for (Index idx = 0; idx < dim * dim; ++idx) {
        result[idx] = jump[idx] - half * (left[idx] + right[idx]);
    }

    return result;
}

} // namespace culindblad
