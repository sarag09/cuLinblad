#include <petsc.h>

int main(int argc, char **argv) {
    // 1. Initialize PETSc and MPI
    PetscCall(PetscInitialize(&argc, &argv, NULL, "Help message for cuLindblad.\n"));

    // 2. Print a message safely across the cluster
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cuLindblad: PETSc Successfully Initialized!\n"));

    // 3. Clean up memory and shut down MPI
    PetscCall(PetscFinalize());
    return 0;
}