#include <iostream>
#include <vector>

#include "culindblad/backend.hpp"
#include "culindblad/model.hpp"
#include "culindblad/operator_term.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/state_layout.hpp"
#include "culindblad/types.hpp"

int main()
{
    using namespace culindblad;

    std::vector<Complex> full_H1(27 * 27, Complex{0.0, 0.0});
    full_H1[0 * 27 + 0] = Complex{1.0, 0.0};
    full_H1[1 * 27 + 1] = Complex{-1.0, 0.0};

    std::vector<Complex> full_H2(27 * 27, Complex{0.0, 0.0});
    full_H2[0 * 27 + 0] = Complex{0.5, 0.0};
    full_H2[1 * 27 + 1] = Complex{-0.5, 0.0};

    OperatorTerm term1{
        TermKind::Hamiltonian,
        "example_full_hamiltonian_1",
        {0, 1, 2},
        full_H1,
        27,
        27
    };

    OperatorTerm term2{
        TermKind::Hamiltonian,
        "example_full_hamiltonian_2",
        {0, 1, 2},
        full_H2,
        27,
        27
    };

    Model model{
        {3, 3, 3},
        {term1, term2},
        {}
    };

    Solver solver = make_solver(model);

    std::cout << "cuLindblad smoke test" << std::endl;
    std::cout << "Hilbert dimension: " << solver.layout.hilbert_dim << std::endl;
    std::cout << "Density dimension: " << solver.layout.density_dim << std::endl;

    std::cout << "Ket strides: ";
    for (Index stride : solver.layout.ket_strides) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;

    // Build a test density matrix with only rho_(0,1) = 1.
    std::vector<Complex> rho_in(solver.layout.density_dim, Complex{0.0, 0.0});
    std::vector<Complex> rho_out;

    rho_in[0 * solver.layout.hilbert_dim + 1] = Complex{1.0, 0.0};

    apply_liouvillian(solver, rho_in, rho_out);

    std::cout << "Input vector size: " << rho_in.size() << std::endl;
    std::cout << "Output vector size: " << rho_out.size() << std::endl;
    std::cout << "Liouvillian output entry (0,1): "
              << rho_out.at(0 * solver.layout.hilbert_dim + 1) << std::endl;

    return 0;
}