#include <iostream>
#include <vector>
#include <petscvec.h>

#include "culindblad/backend.hpp"
#include "culindblad/model.hpp"
#include "culindblad/operator_term.hpp"
#include "culindblad/liouvillian_terms.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/state_layout.hpp"
#include "culindblad/types.hpp"
#include "culindblad/petsc_apply.hpp"
#include "culindblad/petsc_shell.hpp"

int main(int argc, char** argv)
{
    using namespace culindblad;
    PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
    if (ierr != 0) {
        std::cerr << "PetscInitialize failed." << std::endl;
        return 1;
    }

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

    std::vector<Complex> full_L(27 * 27, Complex{0.0, 0.0});
    full_L[0 * 27 + 1] = Complex{1.0, 0.0};

    OperatorTerm dissipator_term{
        TermKind::Dissipator,
        "example_full_dissipator",
        {0, 1, 2},
        full_L,
        27,
        27
    };

    Model model{
        {3, 3, 3},
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

    // Build a test density matrix with only rho_(0,1) = 1.
    std::vector<Complex> rho_in(solver.layout.density_dim, Complex{0.0, 0.0});
    std::vector<Complex> rho_out(solver.layout.density_dim, Complex{0.0, 0.0});

    rho_in[0 * solver.layout.hilbert_dim + 1] = Complex{1.0, 0.0};

    ConstStateBuffer in_buf{rho_in.data(), rho_in.size()};
    StateBuffer out_buf{rho_out.data(), rho_out.size()};

    apply_liouvillian(solver, in_buf, out_buf);

    std::cout << "Input vector size: " << rho_in.size() << std::endl;
    std::cout << "Output vector size: " << rho_out.size() << std::endl;
    std::cout << "Liouvillian output entry (0,1): "
              << rho_out.at(0 * solver.layout.hilbert_dim + 1) << std::endl;
    std::cout << "Liouvillian output entry (0,0): "
              << rho_out.at(0 * solver.layout.hilbert_dim + 0) << std::endl;  
              
    Vec x = nullptr;
    Vec y = nullptr;

    ierr = VecCreateSeq(PETSC_COMM_SELF, solver.layout.density_dim, &x);
    if (ierr != 0) {
        std::cerr << "VecCreateSeq for x failed." << std::endl;
        PetscFinalize();
        return 1;
    }

    ierr = VecCreateSeq(PETSC_COMM_SELF, solver.layout.density_dim, &y);
    if (ierr != 0) {
        std::cerr << "VecCreateSeq for y failed." << std::endl;
        VecDestroy(&x);
        PetscFinalize();
        return 1;
    }

    ierr = VecSet(x, 0.0);
    if (ierr != 0) {
        std::cerr << "VecSet for x failed." << std::endl;
        VecDestroy(&x);
        VecDestroy(&y);
        PetscFinalize();
        return 1;
    }

    ierr = VecSet(y, 0.0);
    if (ierr != 0) {
        std::cerr << "VecSet for y failed." << std::endl;
        VecDestroy(&x);
        VecDestroy(&y);
        PetscFinalize();
        return 1;
    }

    PetscScalar* x_ptr = nullptr;
    ierr = VecGetArray(x, &x_ptr);
    if (ierr != 0) {
        std::cerr << "VecGetArray for x failed." << std::endl;
        VecDestroy(&x);
        VecDestroy(&y);
        PetscFinalize();
        return 1;
    }

    x_ptr[0 * solver.layout.hilbert_dim + 1] = PetscScalar(1.0);

    ierr = VecRestoreArray(x, &x_ptr);
    if (ierr != 0) {
        std::cerr << "VecRestoreArray for x failed." << std::endl;
        VecDestroy(&x);
        VecDestroy(&y);
        PetscFinalize();
        return 1;
    }

    ierr = apply_liouvillian_vec(solver, x, y);
    if (ierr != 0) {
        std::cerr << "apply_liouvillian_vec failed." << std::endl;
        VecDestroy(&x);
        VecDestroy(&y);
        PetscFinalize();
        return 1;
    }

    PetscScalar* y_ptr = nullptr;
    ierr = VecGetArray(y, &y_ptr);
    if (ierr != 0) {
        std::cerr << "VecGetArray for y failed." << std::endl;
        VecDestroy(&x);
        VecDestroy(&y);
        PetscFinalize();
        return 1;
    }

    std::cout << "PETSc Vec output entry (0,1): "
              << reinterpret_cast<Complex*>(y_ptr)[0 * solver.layout.hilbert_dim + 1]
              << std::endl;

    ierr = VecRestoreArray(y, &y_ptr);
    if (ierr != 0) {
        std::cerr << "VecRestoreArray for y failed." << std::endl;
    }

    Vec y_shell = nullptr;
    Mat L_shell = nullptr;

    ierr = VecCreateSeq(PETSC_COMM_SELF, solver.layout.density_dim, &y_shell);
    if (ierr != 0) {
        std::cerr << "VecCreateSeq for y_shell failed." << std::endl;
        VecRestoreArray(y, &y_ptr);
        VecDestroy(&x);
        VecDestroy(&y);
        PetscFinalize();
        return 1;
    }

    ierr = VecSet(y_shell, 0.0);
    if (ierr != 0) {
        std::cerr << "VecSet for y_shell failed." << std::endl;
        VecRestoreArray(y, &y_ptr);
        VecDestroy(&x);
        VecDestroy(&y);
        VecDestroy(&y_shell);
        PetscFinalize();
        return 1;
    }

    ierr = MatCreateShell(
        PETSC_COMM_SELF,
        solver.layout.density_dim,
        solver.layout.density_dim,
        solver.layout.density_dim,
        solver.layout.density_dim,
        &solver,
        &L_shell);
    if (ierr != 0) {
        std::cerr << "MatCreateShell failed." << std::endl;
        VecRestoreArray(y, &y_ptr);
        VecDestroy(&x);
        VecDestroy(&y);
        VecDestroy(&y_shell);
        PetscFinalize();
        return 1;
    }

    ierr = MatShellSetOperation(
        L_shell,
        MATOP_MULT,
        reinterpret_cast<void (*)()>(matshell_apply));
    if (ierr != 0) {
        std::cerr << "MatShellSetOperation failed." << std::endl;
        VecRestoreArray(y, &y_ptr);
        VecDestroy(&x);
        VecDestroy(&y);
        VecDestroy(&y_shell);
        MatDestroy(&L_shell);
        PetscFinalize();
        return 1;
    }

    ierr = MatMult(L_shell, x, y_shell);
    if (ierr != 0) {
        std::cerr << "MatMult on shell matrix failed." << std::endl;
        VecRestoreArray(y, &y_ptr);
        VecDestroy(&x);
        VecDestroy(&y);
        VecDestroy(&y_shell);
        MatDestroy(&L_shell);
        PetscFinalize();
        return 1;
    }

    PetscScalar* y_shell_ptr = nullptr;
    ierr = VecGetArray(y_shell, &y_shell_ptr);
    if (ierr != 0) {
        std::cerr << "VecGetArray for y_shell failed." << std::endl;
        VecRestoreArray(y, &y_ptr);
        VecDestroy(&x);
        VecDestroy(&y);
        VecDestroy(&y_shell);
        MatDestroy(&L_shell);
        PetscFinalize();
        return 1;
    }

    std::cout << "MatShell output entry (0,1): "
              << reinterpret_cast<Complex*>(y_shell_ptr)[0 * solver.layout.hilbert_dim + 1]
              << std::endl;

    ierr = VecRestoreArray(y_shell, &y_shell_ptr);
    if (ierr != 0) {
        std::cerr << "VecRestoreArray for y_shell failed." << std::endl;
    }

    MatDestroy(&L_shell);
    VecDestroy(&y_shell);    

    VecDestroy(&x);
    VecDestroy(&y);            
              
    PetscFinalize();          

    return 0;
}