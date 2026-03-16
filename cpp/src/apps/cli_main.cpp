#include <iostream>
#include <vector>

#include "culindblad/types.hpp"
#include "culindblad/operator_term.hpp"
#include "culindblad/model.hpp"
#include "culindblad/local_dims.hpp"
#include "culindblad/state_layout.hpp"
#include "culindblad/solver.hpp"
#include "culindblad/backend.hpp"

int main()
{
    using namespace culindblad;

    std::cout << "cuLindblad core solver starting..." << std::endl;

    Real r = 1.23;
    Complex z = {2.0, -1.0};
    Index n = 42;

    std::cout << "Real example: " << r << std::endl;
    std::cout << "Complex example: " << z << std::endl;
    std::cout << "Index example: " << n << std::endl;

    OperatorTerm term{
        TermKind::Hamiltonian,
        "example_qutrit_term",
        {0},
        {
            Complex{1.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0},
            Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0},
            Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{-1.0, 0.0}
        },
        3,
        3
    };

    std::cout << "Term name: " << term.name << std::endl;
    std::cout << "Term kind: "
              << (term.kind == TermKind::Hamiltonian ? "Hamiltonian" : "Dissipator")
              << std::endl;
    std::cout << "Number of sites: " << term.sites.size() << std::endl;
    std::cout << "First site: " << term.sites.at(0) << std::endl;
    std::cout << "Matrix rows: " << term.row_dim << std::endl;
    std::cout << "Matrix cols: " << term.col_dim << std::endl;
    std::cout << "Stored matrix entries: " << term.matrix.size() << std::endl;

    Model model{
        {3, 3, 3},
        {term},
        {}
    };

    Solver solver = make_solver(model);

    std::cout << "Solver created successfully." << std::endl;

    std::cout << "Solver Hilbert dimension: "
            << solver.layout.hilbert_dim << std::endl;

    std::cout << "Solver density dimension: "
            << solver.layout.density_dim << std::endl;

    Index subsystems = num_subsystems(model.local_dims);
    Index hilbert_dim = total_hilbert_dim(model.local_dims);

    std::cout << "Subsystem count (helper): " << subsystems << std::endl;
    std::cout << "Total Hilbert dimension: " << hilbert_dim << std::endl;

    std::cout << "Model subsystem count: " << model.local_dims.size() << std::endl;
    std::cout << "Model first local dim: " << model.local_dims.at(0) << std::endl;
    std::cout << "Hamiltonian term count: " << model.hamiltonian_terms.size() << std::endl;
    std::cout << "Dissipator term count: " << model.dissipator_terms.size() << std::endl;

    StateLayout layout = make_state_layout(model.local_dims);

    std::cout << "Layout Hilbert dimension: " << layout.hilbert_dim << std::endl;
    std::cout << "Layout density dimension: " << layout.density_dim << std::endl;

    std::cout << "Ket strides: ";
    for (Index stride : layout.ket_strides) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;

    std::cout << "Bra strides: ";
    for (Index stride : layout.bra_strides) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;

    std::vector<Index> example_state = {1,0,2};

    Index flat_index = flatten_ket_index(example_state, layout.ket_strides);

    std::cout << "Example state (1,0,2) flattened index: "
              << flat_index << std::endl;

    std::vector<Index> recovered =
    unflatten_ket_index(flat_index, layout.ket_strides, layout.local_dims);

    std::cout << "Recovered state: ";

    for (Index x : recovered) {
        std::cout << x << " ";
    }

    std::cout << std::endl;     

    Index example_bra_index = 4;
    Index density_flat =
        flatten_density_index(flat_index, example_bra_index, layout.hilbert_dim);

    std::cout << "Example density index (ket=11, bra=4) flattened: "
              << density_flat << std::endl;

    auto recovered_density =
        unflatten_density_index(density_flat, layout.hilbert_dim);

    std::cout << "Recovered density indices: "
              << "ket=" << recovered_density.first
              << ", bra=" << recovered_density.second
              << std::endl;

    std::vector<Complex> rho_in(solver.layout.density_dim, Complex{0.0, 0.0});
    std::vector<Complex> rho_out;

    rho_in[0] = Complex{1.0, 0.0};

    apply_liouvillian(solver, rho_in, rho_out);

    std::cout << "Input vector size: " << rho_in.size() << std::endl;
    std::cout << "Output vector size: " << rho_out.size() << std::endl;
    std::cout << "First output entry: " << rho_out.at(0) << std::endl;

    return 0;
}
