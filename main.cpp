#include <petsc.h>
#include <complex>
#include <pybind11/pybind11.h>

// Physics content
typedef struct {
    double coupling;
    double gamma;
} AppCtx;


// RHS Function - Matrix free Linblad equation
PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec rho, Vec rho_dot, void *ctx) {
    PetscFunctionBeginUser;
    AppCtx *user = (AppCtx*)ctx;

    // Get raw memory array
    const PetscScalar *rho_raw;
    PetscScalar *dot_raw;
    PetscCall(VecGetArrayRead(rho, &rho_raw));
    PetscCall(VecGetArray(rho_dot, &dot_raw));

    // Cast raw array into complex no.
    auto *rho_c = reinterpret_cast<const std::complex<double>*>(rho_raw);
    auto *dot_c = reinterpret_cast<std::complex<double>*>(dot_raw);

    // Derivative array  = 0
    for(int i = 0; i < 16; i++) {
        dot_c[i] = std::complex<double>(0.0, 0.0);
    }

    // Matrix-Free Hamiltonian: -i [H, rho] = -i(H*rho - rho*H)
    // H only connects index 1 (|01>) and index 2 (|10>) with a value of 2*coupling

    double h_val = 2.0 * user->coupling;
    std::complex<double> neg_i(0.0, -1.0);

    for(int j=0; j<4; j++) {
        // H*rho
        dot_c[1*4 + j] += neg_i * h_val * rho_c[2*4 + j]; // H[1,2] = 2*coupling
        dot_c[2*4 + j] += neg_i * h_val * rho_c[1*4 + j]; // H[2,1] = 2*coupling

        // rho*H
        dot_c[j*4 + 1] -= neg_i * h_val * rho_c[j*4 + 2]; // H[1,2] = 2*coupling
        dot_c[j*4 + 2] -= neg_i * h_val * rho_c[j*4 + 1]; // H[2,1] = 2*coupling
    }

    // Matrix-Free Decay: L * rho * L^dagger - 0.5 * {L^dagger * L, rho}
    // L moves population from index 2 to 0, and from 3 to 1.
    double gamma = user->gamma;

    // Jump term: L * rho * L^dagger (Adds population to lower states)
    dot_c[0*4 + 0] += gamma * rho_c[2*4 + 2]; // |10><10| decays to |00><00|
    dot_c[1*4 + 1] += gamma * rho_c[3*4 + 3]; // |11><11| decays to |01><01|

    // Anti-commutator: -0.5 * (L^dagger L * rho + rho * L^dagger L)

    for(int j=0; j<4; j++) {
        // L^dagger L * rho (Removes population from upper states)
        dot_c[2*4 + j] -= 0.5 * gamma * rho_c[2*4 + j]; // |10><10| loses population
        dot_c[3*4 + j] -= 0.5 * gamma * rho_c[3*4 + j]; // |11><11| loses population

        // rho * L^dagger L (Same as above, but on the right)
        dot_c[j*4 + 2] -= 0.5 * gamma * rho_c[j*4 + 2]; // |10><10| loses population
        dot_c[j*4 + 3] -= 0.5 * gamma * rho_c[j*4 + 3]; // |11><11| loses population
    }

    // Restore arrays
    PetscCall(VecRestoreArrayRead(rho, &rho_raw));
    PetscCall(VecRestoreArray(rho_dot, &dot_raw));
    PetscFunctionReturn(PETSC_SUCCESS);

}

// Replaced int main with callable function
double run_simulation(double coupling_strength, double gamma_rate) {
    // Initialize PETSc with command line arguments
    PetscCallAbort(PETSC_COMM_WORLD, PetscInitializeNoArguments());

    // Create AppCtx and set parameters
    AppCtx user;
    user.coupling = coupling_strength; // Coupling strength
    user.gamma = gamma_rate;    // Decay rate

    // Create density matrix vector (16 elements for 4x4 matrix)
    Vec rho;
    PetscCallAbort(PETSC_COMM_WORLD, VecCreate(PETSC_COMM_WORLD, &rho));
    PetscCallAbort(PETSC_COMM_WORLD, VecSetSizes(rho, PETSC_DECIDE, 32)); // 16 complex numbers = 32 real numbers
    PetscCallAbort(PETSC_COMM_WORLD, VecSetFromOptions(rho));
    PetscCallAbort(PETSC_COMM_WORLD, VecSetUp(rho));

    // Set initial state: |10><10| (index 2 has population) - index = 20
    PetscInt inital_index = 2*(2*4 + 2); // |10><10| element
    PetscScalar initial_value = 1.0; // Population of 1 in |10><10|
    PetscCallAbort(PETSC_COMM_WORLD, VecSetValue(rho, inital_index, initial_value, INSERT_VALUES));
    PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyBegin(rho));
    PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyEnd(rho));
    

    // Create TS solver
    TS ts;
    PetscCallAbort(PETSC_COMM_WORLD, TSCreate(PETSC_COMM_WORLD, &ts));
    PetscCallAbort(PETSC_COMM_WORLD, TSSetRHSFunction(ts, NULL, FormRHSFunction, &user));
    PetscCallAbort(PETSC_COMM_WORLD, TSSetTime(ts, 0.0)); // Initial time
    PetscCallAbort(PETSC_COMM_WORLD, TSSetMaxTime(ts, 1.0)); // Final time
    PetscCallAbort(PETSC_COMM_WORLD, TSSetTimeStep(ts, 0.01)); // Time step size
    PetscCallAbort(PETSC_COMM_WORLD, TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP)); // Match final time to last step
    PetscCallAbort(PETSC_COMM_WORLD, TSSetFromOptions(ts));

    // Hardcode Runge-Kutta since we don't have command line arguments anymore
    PetscCallAbort(PETSC_COMM_WORLD, TSSetType(ts, TSRK));

    // Solve the system
    PetscCallAbort(PETSC_COMM_WORLD, TSSolve(ts, rho));

    // Extract the final results
    const PetscScalar *final_rho;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(rho, &final_rho));
    double final_population = final_rho[0]; // Extract population of |00> (index 0)
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(rho, &final_rho));

    // Clean up
    PetscCallAbort(PETSC_COMM_WORLD, TSDestroy(&ts));
    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&rho));
    PetscCallAbort(PETSC_COMM_WORLD, PetscFinalize());
    return final_population;
}

// Pybind11 module definition
namespace py = pybind11;
PYBIND11_MODULE(cuLindblad_core, m) {
    m.doc() = "cuLindblad core C++ engine via PETSc";
    m.def("run_simulation", &run_simulation, "Run the Lindblad simulation with given coupling strength and gamma rate",
          py::arg("coupling_strength"), py::arg("gamma_rate"));
}