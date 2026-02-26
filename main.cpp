#include <petsc.h>

int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, "cuLindblad Step 3\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cuLindblad: Initializing Density Matrix...\n"));

    // 1. Declare the PETSc Vector pointer
    Vec rho;
    
    // 2. Create the vector attached to the global MPI network
    PetscCall(VecCreate(PETSC_COMM_WORLD, &rho));
    
    // 3. Set the size (2-qubit density matrix = 4x4 = 16 elements)
    PetscInt n_states = 4;
    PetscInt dim = n_states * n_states; 
    
    // Tell PETSc: "Figure out how to split this among CPUs automatically, but the total is 'dim'"
    PetscCall(VecSetSizes(rho, PETSC_DECIDE, dim));
    
    // 4. Allow command-line overrides (Crucial for HPC!)
    PetscCall(VecSetFromOptions(rho));
    
    // 5. Actually allocate the RAM
    PetscCall(VecSetUp(rho));

    // 6. Print the vector to the screen so we can see it
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Density Matrix created with %d elements. Default values:\n", dim));
    PetscCall(VecView(rho, PETSC_VIEWER_STDOUT_WORLD));

    // 7. Clean up memory (No memory leaks allowed!)
    PetscCall(VecDestroy(&rho));
    PetscCall(PetscFinalize());
    return 0;
}