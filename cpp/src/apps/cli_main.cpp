#include <iostream>
#include <vector>

#include "culindblad/types.hpp"
#include "culindblad/operator_term.hpp"
#include "culindblad/model.hpp"
#include "culindblad/local_dims.hpp"

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

    Index subsystems = num_subsystems(model.local_dims);
    Index hilbert_dim = total_hilbert_dim(model.local_dims);

    std::cout << "Subsystem count (helper): " << subsystems << std::endl;
    std::cout << "Total Hilbert dimension: " << hilbert_dim << std::endl;

    std::cout << "Model subsystem count: " << model.local_dims.size() << std::endl;
    std::cout << "Model first local dim: " << model.local_dims.at(0) << std::endl;
    std::cout << "Hamiltonian term count: " << model.hamiltonian_terms.size() << std::endl;
    std::cout << "Dissipator term count: " << model.dissipator_terms.size() << std::endl;

    return 0;
}
