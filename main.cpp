#include <petsc.h>

// Physics content
typedef struct {
    PetscReal coupling_strength;
} AppCtx;

// Matrix free muliplication function
PetscErrorCode MyMatrixFreeMultiply(Mat H, Vec rho, Vec rho_dot) {
    PetscFunctionBeginUser;

    // Unpack physics parameters
    AppCtx *user;
    PetscCall(MatShellGetContext(H, &user));

    // Get row arrays
    const PetscScalar *rho_array;
    PetscScalar *rho_dot_array;
    PetscCall(VecGetArrayRead(rho, &rho_array));
    PetscCall(VecGetArray(rho_dot, &rho_dot_array));

    // Swaps state 0 and 1 with coupling strength
    rho_dot_array[0] = user->coupling_strength * rho_array[1];
    rho_dot_array[1] = user->coupling_strength * rho_array[0];

    // Set the rest to zero
    for (int i = 2; i < 16; i++) {
        rho_dot_array[i] = 0.0;
    }

    // Restore arrays
    PetscCall(VecRestoreArrayRead(rho, &rho_array));
    PetscCall(VecRestoreArray(rho_dot, &rho_dot_array));

    PetscFunctionReturn(PETSC_SUCCESS);
}


// RHS Function
PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec rho, Vec rho_dot, void *ctx) {
    PetscFunctionBeginUser;

    // Pass Matrix H to ctx
    Mat H = (Mat)ctx;

    // H is a "Shell", this will call MyMatrixFreeMultiply
    PetscCall(MatMult(H, rho, rho_dot));
    
    PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, "Matrix free Demo - Step 6\n"));

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


    // coupling strength for the Hamiltonian
    AppCtx user;
    user.coupling_strength = 2; 

    // Create shell matrix for the Hamiltonian
    Mat H;
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, dim, dim, &user, &H));
    PetscCall(MatShellSetOperation(H, MATOP_MULT, (void (*)(void))MyMatrixFreeMultiply));

    // TS Setup
    TS ts;
    PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
    PetscCall(TSSetRHSFunction(ts, NULL, FormRHSFunction, H));
    PetscCall(TSSetTime(ts, 0.0));
    PetscCall(TSSetMaxTime(ts, 1.0));
    PetscCall(TSSetTimeStep(ts, 0.1));
    PetscCall(TSSetFromOptions(ts));

    // Run Simulation
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Starting ODE Solve...\n"));
    PetscCall(TSSolve(ts,rho));

    // View the final result
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final Density Matrix at t=1.0:\n"));
    PetscCall(VecView(rho, PETSC_VIEWER_STDOUT_WORLD));

    //  Clean up memory (No memory leaks allowed!)
    PetscCall(TSDestroy(&ts));
    PetscCall(VecDestroy(&rho));
    PetscCall(MatDestroy(&H));
    PetscCall(PetscFinalize());
    return 0;
}