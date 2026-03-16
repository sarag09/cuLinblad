#include <stdexcept>
#include <vector>

#include "culindblad/cpu_reference.hpp"
#include "culindblad/liouvillian_terms.hpp"

namespace culindblad {

std::vector<Complex> apply_hamiltonian_commutator(
    const std::vector<Complex>& H,
    const std::vector<Complex>& rho,
    Index dim)
{
    if (H.size() != dim * dim) {
        throw std::runtime_error("apply_hamiltonian_commutator: H has wrong size");
    }

    if (rho.size() != dim * dim) {
        throw std::runtime_error("apply_hamiltonian_commutator: rho has wrong size");
    }

    const std::vector<Complex> H_rho = multiply_square_matrices(H, rho, dim);
    const std::vector<Complex> rho_H = multiply_square_matrices(rho, H, dim);

    std::vector<Complex> result(dim * dim, Complex{0.0, 0.0});
    const Complex minus_i{0.0, -1.0};

    for (Index idx = 0; idx < dim * dim; ++idx) {
        result[idx] = minus_i * (H_rho[idx] - rho_H[idx]);
    }

    return result;
}

} // namespace culindblad
