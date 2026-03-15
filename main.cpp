#include <petsc.h>
#include <petscts.h>
 
// CUDA-backend PETSc helpers (this is where PetscGetCurrentCUDAStream lives)
#include <petscdevice_cuda.h>   // PetscGetCurrentCUDAStream, PetscCUBLASGetHandle, PetscCallCUBLAS
 
#include <cublas_v2.h>
#include <cuda_runtime.h>
 
#include <vector>
#include <complex>
 
// Application context to hold parameters
typedef struct {
  PetscInt        N;            // Hilbert space dimension
  cublasHandle_t  cublas_handle;
  cuDoubleComplex *d_H;         // Hamiltonian
  cuDoubleComplex *d_L;         // Jump operator L
  cuDoubleComplex *d_LdagL;     // L^\dagger L
  cuDoubleComplex *d_temp;      // workspace
  double          gamma;        // dissipation rate
} AppCtx;
 
// Matrix-free Lindblad action: rho_dot = L(rho)
PetscErrorCode ComputeLiouvillianAction(Mat L, Vec rho_in, Vec rho_dot_out)
{
  AppCtx *ctx = NULL;
  PetscCall(MatShellGetContext(L, &ctx));
 
  const PetscInt N = ctx->N;
 
  const PetscScalar *d_rho     = NULL;
  PetscScalar       *d_rho_dot = NULL;
 
  PetscCall(VecCUDAGetArrayRead(rho_in, &d_rho));
  PetscCall(VecCUDAGetArrayWrite(rho_dot_out, &d_rho_dot));
 
  const cuDoubleComplex *d_rho_cu     = (const cuDoubleComplex*)d_rho;
  cuDoubleComplex       *d_rho_dot_cu = (cuDoubleComplex*)d_rho_dot;
 
  // Sync cuBLAS to PETSc's current CUDA stream
  cudaStream_t stream = 0;
  PetscCall(PetscGetCurrentCUDAStream(&stream));
  PetscCallCUBLAS(cublasSetStream(ctx->cublas_handle, stream));
  PetscCallCUBLAS(cublasSetPointerMode(ctx->cublas_handle, CUBLAS_POINTER_MODE_HOST));
 
  // cuBLAS constants (host pointers since POINTER_MODE_HOST)
  const cuDoubleComplex minus_i         = make_cuDoubleComplex(0.0, -1.0);
  const cuDoubleComplex plus_i          = make_cuDoubleComplex(0.0,  1.0);
  const cuDoubleComplex zero            = make_cuDoubleComplex(0.0,  0.0);
  const cuDoubleComplex one             = make_cuDoubleComplex(1.0,  0.0);
  const cuDoubleComplex minus_half_gamma= make_cuDoubleComplex(-0.5 * ctx->gamma, 0.0);
  const cuDoubleComplex plus_gamma      = make_cuDoubleComplex( ctx->gamma, 0.0);
 
  // Unitary part: -i[H,rho] = -i(H*rho - rho*H)
  // rho_dot = -i H rho
  PetscCallCUBLAS(cublasZgemm(ctx->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, N, N,
&minus_i,
                              ctx->d_H, N,
                              d_rho_cu, N,
&zero,
                              d_rho_dot_cu, N));
  // rho_dot += +i rho H
  PetscCallCUBLAS(cublasZgemm(ctx->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, N, N,
&plus_i,
                              d_rho_cu, N,
                              ctx->d_H, N,
&one,
                              d_rho_dot_cu, N));
 
  // Dissipator: gamma ( L rho L^\dagger - 1/2 {L^\dagger L, rho} )
  // rho_dot -= 0.5 gamma (L^\dagger L) rho
  PetscCallCUBLAS(cublasZgemm(ctx->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, N, N,
&minus_half_gamma,
                              ctx->d_LdagL, N,
                              d_rho_cu, N,
&one,
                              d_rho_dot_cu, N));
  // rho_dot -= 0.5 gamma rho (L^\dagger L)
  PetscCallCUBLAS(cublasZgemm(ctx->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, N, N,
&minus_half_gamma,
                              d_rho_cu, N,
                              ctx->d_LdagL, N,
&one,
                              d_rho_dot_cu, N));
 
  // temp = L rho
  PetscCallCUBLAS(cublasZgemm(ctx->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, N, N,
&one,
                              ctx->d_L, N,
                              d_rho_cu, N,
&zero,
                              ctx->d_temp, N));
  // rho_dot += gamma temp L^\dagger  (note: opC on L gives L^\dagger)
  PetscCallCUBLAS(cublasZgemm(ctx->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_C,
                              N, N, N,
&plus_gamma,
                              ctx->d_temp, N,
                              ctx->d_L, N,
&one,
                              d_rho_dot_cu, N));
 
  // Debug/robustness: ensure completion before PETSc uses rho_dot
  PetscCallCUDA(cudaStreamSynchronize(stream));
 
  PetscCall(VecCUDARestoreArrayRead(rho_in, &d_rho));
  PetscCall(VecCUDARestoreArrayWrite(rho_dot_out, &d_rho_dot));
  return PETSC_SUCCESS;
}
 
// RHS function wrapper: rho_dot = L_shell * rho
PetscErrorCode MyRHSFunction(TS ts, PetscReal t, Vec rho_in, Vec rho_dot_out, void *ctx)
{
  Mat L_shell = (Mat)ctx;
  PetscCall(MatMult(L_shell, rho_in, rho_dot_out));
  return PETSC_SUCCESS;
}
 
int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
 
#if !defined(PETSC_HAVE_CUDA)
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This example requires PETSc configured with CUDA");
#endif
 
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This example requires PETSc configured with complex scalars (--with-scalar-type=complex)");
#endif
 
  // ABI sanity checks
  static_assert(sizeof(std::complex<double>) == sizeof(cuDoubleComplex),
                "std::complex<double> must match cuDoubleComplex layout for cudaMemcpy");
 
  PetscCheck(sizeof(PetscScalar) == sizeof(cuDoubleComplex),
             PETSC_COMM_WORLD, PETSC_ERR_SUP,
             "PetscScalar must match cuDoubleComplex size (complex double PETSc build required)");
 
  AppCtx ctx;
  ctx.N     = 4;
  ctx.gamma = 1.0;
 
  const PetscInt N_sq = ctx.N * ctx.N;
 
  // Use PETSc-provided cuBLAS handle for consistency with PETSc CUDA setup
  PetscCall(PetscCUBLASGetHandle(&ctx.cublas_handle));
 
  // Host matrices (column-major for cuBLAS)
  std::vector<std::complex<double>> h_H(N_sq,     {0.0, 0.0});
  std::vector<std::complex<double>> h_L(N_sq,     {0.0, 0.0});
  std::vector<std::complex<double>> h_LdagL(N_sq, {0.0, 0.0});
 
  // L = |00><01| : row 0, col 1 => index = col*N + row = 1*4 + 0 = 4
  h_L[4] = {1.0, 0.0};
  // L^\dagger L = |01><01| : row 1, col 1 => index = 1*4 + 1 = 5
  h_LdagL[5] = {1.0, 0.0};
 
  // Allocate device matrices
  PetscCallCUDA(cudaMalloc((void**)&ctx.d_H,     N_sq * sizeof(cuDoubleComplex)));
  PetscCallCUDA(cudaMalloc((void**)&ctx.d_L,     N_sq * sizeof(cuDoubleComplex)));
  PetscCallCUDA(cudaMalloc((void**)&ctx.d_LdagL, N_sq * sizeof(cuDoubleComplex)));
  PetscCallCUDA(cudaMalloc((void**)&ctx.d_temp,  N_sq * sizeof(cuDoubleComplex)));
 
  PetscCallCUDA(cudaMemcpy(ctx.d_H,     h_H.data(),     N_sq * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  PetscCallCUDA(cudaMemcpy(ctx.d_L,     h_L.data(),     N_sq * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  PetscCallCUDA(cudaMemcpy(ctx.d_LdagL, h_LdagL.data(), N_sq * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
 
  // Shell matrix representing Liouvillian action
  Mat L_shell;
  PetscCall(MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N_sq, N_sq, &ctx, &L_shell));
  PetscCall(MatShellSetOperation(L_shell, MATOP_MULT, (void(*)(void))ComputeLiouvillianAction));
 
  // Force MatCreateVecs() for this shell matrix to produce CUDA vectors
  PetscCall(MatShellSetVecType(L_shell, VECCUDA));
 
  // Initial state rho (stored as vectorized N x N matrix, column-major)
  Vec rho;
  PetscCall(VecCreate(PETSC_COMM_WORLD, &rho));
  PetscCall(VecSetSizes(rho, PETSC_DECIDE, N_sq));
  PetscCall(VecSetType(rho, VECCUDA));
  PetscCall(VecSetFromOptions(rho));
  PetscCall(VecSetUp(rho));
 
  // rho = |01><01| : row 1, col 1 => index 5
  PetscCall(VecSetValue(rho, 5, 1.0, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(rho));
  PetscCall(VecAssemblyEnd(rho));
 
  // Time stepping
  TS ts;
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_LINEAR));
  PetscCall(TSSetType(ts, TSRK));
  PetscCall(TSSetRHSFunction(ts, NULL, MyRHSFunction, (void*)L_shell));
  PetscCall(TSSetTimeStep(ts, 0.01));
  PetscCall(TSSetMaxTime(ts, 1.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetSolution(ts, rho));
  PetscCall(TSSetFromOptions(ts));
 
  PetscCall(TSSolve(ts, rho));
 
  // Print final populations (host-readable view)
  const PetscScalar *rho_final = NULL;
  PetscCall(VecGetArrayRead(rho, &rho_final));
 
  const double pop_00 = (double)PetscRealPart(rho_final[0]); // |00><00| index 0
  const double pop_01 = (double)PetscRealPart(rho_final[5]); // |01><01| index 5
 
  PetscPrintf(PETSC_COMM_WORLD, "\nSimulation Complete!\n");
  PetscPrintf(PETSC_COMM_WORLD, "Population of |00> at t=1.0: %g  (Expected ~0.63212)\n", pop_00);
  PetscPrintf(PETSC_COMM_WORLD, "Population of |01> at t=1.0: %g  (Expected ~0.36788)\n\n", pop_01);
 
  PetscCall(VecRestoreArrayRead(rho, &rho_final));
 
  // Cleanup
  PetscCall(MatDestroy(&L_shell));
  PetscCall(VecDestroy(&rho));
 
  PetscCallCUDA(cudaFree(ctx.d_H));
  PetscCallCUDA(cudaFree(ctx.d_L));
  PetscCallCUDA(cudaFree(ctx.d_LdagL));
  PetscCallCUDA(cudaFree(ctx.d_temp));
 
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}