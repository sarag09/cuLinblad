#include <iostream>
#include <vector>

#include <petscts.h>

#include "culindblad/backend.hpp"
#include "culindblad/local_operator_embed.hpp"
#include "culindblad/model.hpp"
#include "culindblad/operator_term.hpp"
#include "culindblad/petsc_ts_smoke.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/types.hpp"
#include "culindblad/local_apply.hpp"
#include "culindblad/liouvillian_terms.hpp"

int main(int argc, char** argv)
{
    using namespace culindblad;

    PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
    if (ierr != 0) {
        std::cerr << "PetscInitialize failed." << std::endl;
        return 1;
    }

    const std::vector<Index> local_dims = {3, 3, 3};

    std::vector<Complex> z_like = {
        Complex{1.0, 0.0},  Complex{0.0, 0.0},  Complex{0.0, 0.0},
        Complex{0.0, 0.0},  Complex{-1.0, 0.0}, Complex{0.0, 0.0},
        Complex{0.0, 0.0},  Complex{0.0, 0.0},  Complex{0.0, 0.0}
    };

    std::vector<Complex> half_z_like = {
        Complex{0.5, 0.0},  Complex{0.0, 0.0},  Complex{0.0, 0.0},
        Complex{0.0, 0.0},  Complex{-0.5, 0.0}, Complex{0.0, 0.0},
        Complex{0.0, 0.0},  Complex{0.0, 0.0},  Complex{0.0, 0.0}
    };

    std::vector<Complex> lowering = {
        Complex{0.0, 0.0}, Complex{1.0, 0.0}, Complex{0.0, 0.0},
        Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0},
        Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0}
    };

    std::vector<Complex> full_H1 = embed_one_site_operator(z_like, 3, 0, local_dims);
    std::vector<Complex> full_H2 = embed_one_site_operator(half_z_like, 3, 0, local_dims);
    std::vector<Complex> full_L  = embed_one_site_operator(lowering, 3, 0, local_dims);

    const Index hilbert_dim = 27;

    OperatorTerm term1{
        TermKind::Hamiltonian,
        "q1_z_like",
        {0},
        full_H1,
        hilbert_dim,
        hilbert_dim
    };

    OperatorTerm term2{
        TermKind::Hamiltonian,
        "q1_half_z_like",
        {0},
        full_H2,
        hilbert_dim,
        hilbert_dim
    };

    OperatorTerm dissipator_term{
        TermKind::Dissipator,
        "q1_lowering",
        {0},
        full_L,
        hilbert_dim,
        hilbert_dim
    };

    Model model{
        local_dims,
        {term1, term2},
        {dissipator_term}
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

    std::vector<Complex> rho_in(solver.layout.density_dim, Complex{0.0, 0.0});
    std::vector<Complex> rho_out(solver.layout.density_dim, Complex{0.0, 0.0});

    rho_in[0 * solver.layout.hilbert_dim + 9] = Complex{1.0, 0.0};

    ConstStateBuffer in_buf{rho_in.data(), rho_in.size()};
    StateBuffer out_buf{rho_out.data(), rho_out.size()};

    apply_liouvillian(solver, in_buf, out_buf);

    std::cout << "Liouvillian output entry (0,9): "
            << rho_out.at(0 * solver.layout.hilbert_dim + 9) << std::endl;
    std::cout << "Liouvillian output entry (0,0): "
            << rho_out.at(0 * solver.layout.hilbert_dim + 0) << std::endl;

    std::vector<Complex> local_left =
        apply_one_site_operator_left(z_like, 3, 0, local_dims, in_buf);

    std::cout << "Local left-action entry (0,9): "
            << local_left.at(0 * solver.layout.hilbert_dim + 9) << std::endl;    
            
    std::vector<Complex> local_right =
    apply_one_site_operator_right(z_like, 3, 0, local_dims, in_buf);

    std::cout << "Local right-action entry (0,9): "
              << local_right.at(0 * solver.layout.hilbert_dim + 9) << std::endl;  
              
    std::vector<Complex> local_comm =
        apply_one_site_commutator(z_like, 3, 0, local_dims, in_buf);

    std::cout << "Local commutator entry (0,9): "
              << local_comm.at(0 * solver.layout.hilbert_dim + 9) << std::endl;   
              
    std::vector<Complex> embedded_z =
        embed_one_site_operator(z_like, 3, 0, local_dims);

    ConstStateBuffer rho_compare_buf{rho_in.data(), rho_in.size()};

    std::vector<Complex> dense_comm =
        apply_hamiltonian_commutator(
            embedded_z,
            rho_compare_buf,
            solver.layout.hilbert_dim);

    std::cout << "Embedded dense commutator entry (0,9): "
              << dense_comm.at(0 * solver.layout.hilbert_dim + 9) << std::endl;

    std::cout << "Difference local-vs-dense at (0,9): "
              << (local_comm.at(0 * solver.layout.hilbert_dim + 9)
                  - dense_comm.at(0 * solver.layout.hilbert_dim + 9))
              << std::endl;              

    Complex ts_value{0.0, 0.0};
    ierr = run_ts_smoke_test(solver, 0, 9, ts_value);
    if (ierr != 0) {
        std::cerr << "run_ts_smoke_test failed." << std::endl;
        PetscFinalize();
        return 1;
    }

    std::cout << "TS evolved entry (0,9): " << ts_value << std::endl;

    PetscFinalize();
    return 0;
}