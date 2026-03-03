#include <petsc.h>
#include <complex>
#include <pybind11/pybind11.h>
#include <cutensor.h>
#include <cuda_runtime.h>
#include <vector>

// Physics content and gpu tools
typedef struct {
    double coupling;
    double gamma;
    cutensorHandle_t cutensor_handle;
    std::complex<double>* H_rho; // Hamiltonian in gpu
} AppCtx;


// RHS Function - Matrix free Linblad equation
PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec rho, Vec rho_dot, void *ctx) {
    PetscFunctionBeginUser;
    AppCtx *user = (AppCtx*)ctx;

    PetscCall(VecSet(rho_dot, 0.0)); // set output to zero 

    // Get gpu memory array
    const PetscScalar *rho_gpu;
    PetscScalar *dot_gpu;
    PetscCall(VecCUDAGetArrayRead(rho, &rho_gpu));
    PetscCall(VecCUDAGetArray(rho_dot, &dot_gpu));

    // Restore arrays
    PetscCall(VecCUDARestoreArrayRead(rho, &rho_gpu));
    PetscCall(VecCUDARestoreArray(rho_dot, &dot_gpu));
    PetscFunctionReturn(PETSC_SUCCESS);

}

// Replaced int main with callable function
double run_simulation(double coupling_strength, double gamma_rate) {
    // Initialize PETSc with command line arguments
    PetscCallAbort(PETSC_COMM_WORLD, PetscInitializeNoArguments());

    // Initialize cuTENSOR
    cutensorHandle_t cutensor_handle;
    cutensorStatus_t cutensor_status = cutensorCreate(&cutensor_handle);

    if (cutensor_status != CUTENSOR_STATUS_SUCCESS) {
        PetscPrintf(PETSC_COMM_WORLD, "cuTENSOR initialization failed: %s\n", cutensorGetErrorString(cutensor_status));
        return -1.0; // Return error code on failure
    }

    PetscPrintf(PETSC_COMM_WORLD, "cuTENSOR initialized successfully.\n");

    // Create AppCtx and set parameters
    AppCtx user;
    user.coupling = coupling_strength; // Coupling strength
    user.gamma = gamma_rate;    // Decay rate

    // Build Hamiltionian on CPU and copy to GPU
    std::vector<std::complex<double>> H_rho(16, std::complex<double>(0.0, 0.0)); // 4x4 matrix = 16 elements
    H_rho[1*4 + 2] = std::complex<double>(0.0, 2.0 * coupling_strength); // |10><00| element
    H_rho[2*4 + 1] = std::complex<double>(0.0, 2.0 * coupling_strength); // |00><10| element

    // Allocate GPU memory for Hamiltonian and copy data
    cudaMalloc((void**)&user.H_rho, 16 * sizeof(std::complex<double>));
    cudaMemcpy(user.H_rho, H_rho.data(), 16 * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

    // Create density matrix vector (16 elements for 4x4 matrix)
    Vec rho;
    PetscCallAbort(PETSC_COMM_WORLD, VecCreate(PETSC_COMM_WORLD, &rho));
    PetscCallAbort(PETSC_COMM_WORLD, VecSetSizes(rho, PETSC_DECIDE, 32)); // 16 complex numbers = 32 real numbers
    PetscCallAbort(PETSC_COMM_WORLD, VecSetType(rho, VECCUDA)); // Use GPU vector type
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
    cutensorDestroy(cutensor_handle);
    cudaFree(user.H_rho);
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