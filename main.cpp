#include <petsc.h>

// Physics content
typedef struct {
    PetscReal gamma;
} AppCtx;

// RHS Function

PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec rho, Vec rho_dot, void *ctx) {
    PetscFunctionBeginUser;

    // Unpack physics parameters
    AppCtx *user = (AppCtx*)ctx;

    // rho_dot = rho;
    PetscCall(VecCopy(rho, rho_dot));

    PetscScalar *rho_array;
    PetscCall(VecGetArray(rho_dot, &rho_array));

    // rho_dot = -gamma * rho[0]
    rho_array[0] = rho_array[0] * (-user->gamma);

    PetscCall(VecRestoreArray(rho_dot, &rho_array));
    
    PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, "cuLindblad ODE Solver\n"));


    // Declare the PETSc Vector pointer
    Vec rho;
    
    //  Create the vector attached to the global MPI network
    PetscCall(VecCreate(PETSC_COMM_WORLD, &rho));
    
    //  Set the size (2-qubit density matrix = 4x4 = 16 elements)
    PetscInt n_states = 4;
    PetscInt dim = n_states * n_states; 
    
    // Tell PETSc: "Figure out how to split this among CPUs automatically, but the total is 'dim'"
    PetscCall(VecSetSizes(rho, PETSC_DECIDE, dim));
    
    // Allow command-line overrides (Crucial for HPC!)
    PetscCall(VecSetFromOptions(rho));
    
    //  Actually allocate the RAM
    PetscCall(VecSetUp(rho));

    // Initial Condition
    PetscCall(VecSet(rho, 1.0));


    //gamma
    AppCtx user;
    user.gamma = 0.5;

    // TS Setup
    TS ts;
    PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
    PetscCall(TSSetRHSFunction(ts, NULL, FormRHSFunction, &user));
    PetscCall(TSSetTime(ts, 0.0));
    PetscCall(TSSetMaxTime(ts, 5.0));
    PetscCall(TSSetTimeStep(ts, 0.1));
    PetscCall(TSSetFromOptions(ts));

    // Run Simulation
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Starting ODE Solve...\n"));
    PetscCall(TSSolve(ts,rho));

    // View the final result
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final Density Matrix at t=5.0:\n"));
    PetscCall(VecView(rho, PETSC_VIEWER_STDOUT_WORLD));

    //  Clean up memory (No memory leaks allowed!)
    PetscCall(TSDestroy(&ts));
    PetscCall(VecDestroy(&rho));
    PetscCall(PetscFinalize());
    return 0;
}