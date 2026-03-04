#include <petsc.h>
#include <petscts.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <complex>

// Application context to hold parameters 
typedef struct {
    PetscInt N; // Hilbert space dimension 
    cublasHandle_t cublas_handle;
    cuDoubleComplex *d_H; // Hamiltonian 
} AppCtx;


//  Matrix free Linblad equation
// computes rho_dot = -i * (H * rho - rho * H) using cuBLAS
PetscErrorCode ComputeLiouvillianAction(Mat L, Vec rho_in, Vec rho_dot_out) {
    AppCtx *ctx;
    PetscCall(MatShellGetContext(L, &ctx));

    PetscInt N = ctx->N;

    // Get raw gpu pointers to the input and output vectors
    const PetscScalar *d_rho;
    PetscScalar *d_rho_dot;
    PetscCall(VecCUDAGetArrayRead(rho_in, &d_rho));
    PetscCall(VecCUDAGetArrayWrite(rho_dot_out, &d_rho_dot));

    // Cast to cuDoubleComplex for cuBLAS
    const cuDoubleComplex *d_rho_cu = (const cuDoubleComplex*)d_rho;
    cuDoubleComplex *d_rho_dot_cu = (cuDoubleComplex*)d_rho_dot;

    // cuBLAS constants
    cuDoubleComplex minus_i = make_cuDoubleComplex(0.0, -1.0); // -i
    cuDoubleComplex plus_i = make_cuDoubleComplex(0.0, 1.0); // +i
    cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);

    // Compute -i * H * rho
    cublasZgemm(ctx->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &minus_i, ctx->d_H, N, d_rho_cu, N, &zero, d_rho_dot_cu, N);
    // Compute rho_dot += +i * rho * H
    cublasZgemm(ctx->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &plus_i, d_rho_cu, N, ctx->d_H, N, &one, d_rho_dot_cu, N);

    // Restore the GPU arrays
    PetscCall(VecCUDARestoreArrayRead(rho_in, &d_rho));
    PetscCall(VecCUDARestoreArrayWrite(rho_dot_out, &d_rho_dot));

    return PETSC_SUCCESS;
}


int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    AppCtx ctx;
    ctx.N = 4; // Example Hilbert space dimension
    PetscInt N_sq = ctx.N * ctx.N;

    // Initialize cuBLAS
    cublasCreate(&ctx.cublas_handle);

    // Build Hamiltonian on cpu and copy to gpu
    std::vector<std::complex<double>> h_H(N_sq, std::complex<double>(0.0, 0.0));
    h_H[1*4 + 2] = std::complex<double>(4.0, 0.0); // Example Hamiltonian element
    h_H[2*4 + 1] = std::complex<double>(4.0, 0.0); 

    // Allocate and copy Hamiltonian to GPU
    cudaMalloc((void**)&ctx.d_H, N_sq * sizeof(cuDoubleComplex));
    cudaMemcpy(ctx.d_H, h_H.data(), N_sq * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // Create a shell matrix for the Liouvillian
    Mat L_shell;
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N_sq, N_sq, &ctx, &L_shell));
    PetscCall(MatShellSetOperation(L_shell, MATOP_MULT, (void(*)(void))ComputeLiouvillianAction));

    // Setup intial state to |00>
    Vec rho;
    PetscCall(VecCreate(PETSC_COMM_WORLD, &rho));
    PetscCall(VecSetSizes(rho, PETSC_DECIDE, N_sq));
    PetscCall(VecSetType(rho, VECCUDA));
    PetscCall(VecSetFromOptions(rho));
    PetscCall(VecSetUp(rho));

    // |00><00| has index 0 in the vectorized density matrix
    PetscCall(VecSetValue(rho, 0, 1.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(rho));
    PetscCall(VecAssemblyEnd(rho));

    // Time-stepping with TS
    TS ts;
    PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
    PetscCall(TSSetProblemType(ts, TS_LINEAR));
    PetscCall(TSSetRHSFunction(ts, NULL, TSComputeRHSFunctionLinear, NULL));
    PetscCall(TSSetRHSJacobian(ts, L_shell, L_shell, TSComputeRHSJacobianConstant, NULL));
    PetscCall(TSSetTimeStep(ts, 0.01));
    PetscCall(TSSetMaxTime(ts, 1.0));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
    PetscCall(TSSetSolution(ts, rho));
    PetscCall(TSSetFromOptions(ts));

    // Run the time-stepping
    PetscCall(TSSolve(ts, rho));

    // Print the final population of |00><00| ---
    const PetscScalar *rho_final_host;
    PetscCall(VecGetArrayRead(rho, &rho_final_host));
    PetscPrintf(PETSC_COMM_WORLD, "\nSimulation Complete!\n");
    PetscPrintf(PETSC_COMM_WORLD, "Population of |00> at t=1.0: %g + %gi\n\n", 
                (double)PetscRealPart(rho_final_host[0]), 
                (double)PetscImaginaryPart(rho_final_host[0]));
    PetscCall(VecRestoreArrayRead(rho, &rho_final_host));

    // Clean up
    PetscCall(MatDestroy(&L_shell));
    PetscCall(VecDestroy(&rho));
    cudaFree(ctx.d_H);
    PetscCall(cublasDestroy(ctx.cublas_handle));
    PetscCall(TSDestroy(&ts));

    PetscCall(PetscFinalize());
    return 0;
}