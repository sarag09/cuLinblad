#include <petsc.h>
#include <complex>

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

int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, "Matrix-Free Lindblad\n"));

    // Create AppCtx and set parameters
    AppCtx user;
    user.coupling = 2.0; // Coupling strength
    user.gamma = 0.5;    // Decay rate

    // Create density matrix vector (16 elements for 4x4 matrix)
    Vec rho;
    PetscCall(VecCreate(PETSC_COMM_WORLD, &rho));
    PetscCall(VecSetSizes(rho, PETSC_DECIDE, 32)); // 16 complex numbers = 32 real numbers
    PetscCall(VecSetFromOptions(rho));
    PetscCall(VecSetUp(rho));

    // Set initial state: |10><10| (index 2 has population) - index = 20
    PetscInt inital_index = 2*(2*4 + 2); // |10><10| element
    PetscScalar initial_value = 1.0; // Population of 1 in |10><10|
    PetscCall(VecSetValue(rho, inital_index, initial_value, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(rho));
    PetscCall(VecAssemblyEnd(rho));
    

    // Create TS solver
    TS ts;
    PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
    PetscCall(TSSetRHSFunction(ts, NULL, FormRHSFunction, &user));
    PetscCall(TSSetTime(ts, 0.0)); // Initial time
    PetscCall(TSSetMaxTime(ts, 1.0)); // Final time
    PetscCall(TSSetTimeStep(ts, 0.01)); // Time step size
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP)); // Match final time to last step
    PetscCall(TSSetFromOptions(ts));

    // Solve the system
    PetscCall(TSSolve(ts, rho));

    // Print out the final population of |00> (which is at index 0)
    const PetscScalar *final_rho;
    PetscCall(VecGetArrayRead(rho, &final_rho));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n--- RESULTS ---\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cuLindblad rho_00 at t=1.0: %f\n", final_rho[0]));
    PetscCall(VecRestoreArrayRead(rho, &final_rho));

    // Clean up
    PetscCall(TSDestroy(&ts));
    PetscCall(VecDestroy(&rho));
    PetscCall(PetscFinalize());
    return 0;
}