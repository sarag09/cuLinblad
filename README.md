# cuLindblad
 
cuLindblad is a matrix-free solver for open quantum system dynamics based on the Lindblad master equation. The project is aimed at scaling exact density-matrix simulation beyond the point where explicit Liouvillian or superoperator construction becomes impractical. The current implementation focuses on a high-performance C++/PETSc/CUDA backend with a validated GPU execution path and structured fast paths for local operator application.
 
The long-term goal is not only to make Lindblad simulation faster, but to generate physically realistic artifacts such as reduced process matrices and effective error channels that can be consumed by higher layers of the quantum-computing stack for resource-estimation and error-correction studies.
 
## Current Status
 
The current repository contains:
 
- a validated matrix-free PETSc time-evolution workflow
- a batched GPU execution path using grouped tensor layouts
- stress-test and validation CLI modes
- structured GPU fast paths for several important operator families
 
The current benchmarked path preserves a trusted `N=2`, `d=2` validation fixture while supporting larger qutrit stress-test runs.
 
The current fixed-step validation configuration uses:
 
- `TSRK`
- `TSRK4`
- `TSADAPTNONE`
 
## Current Capabilities
 
The current implementation supports:
 
- matrix-free Lindblad RHS evaluation
- grouped-layout GPU application of local Hamiltonian and dissipator terms
- batched GPU evolution for multiple density-matrix states
- grouped scratch-buffer reuse to reduce persistent memory footprint
- executor reuse by site layout to reduce GPU residency
- structured operator fast paths including:
  - same-layout static Hamiltonian aggregation
  - same-layout dissipator anti-commutator aggregation
  - diagonal dissipator jump kernels
  - sparse static Hamiltonian commutator kernels
  - diagonal Hamiltonian commutator kernels
 
## Repository Layout
 
```text
cpp/        C++ source code, CUDA kernels, PETSc integration, CLI app
docs/       project documents and supporting writeups
examples/   example workflows and auxiliary material
profiling/  profiling notes and benchmark summaries
tests/      validation and test-related code
build/      local build directory (generated, not for version control)
```
 
## Build Requirements
 
You need a PETSc installation configured for:
 
- complex scalars
- CUDA support if you want the GPU path
- C++17
 
The current code also relies on:
 
- CMake
- a C++17 compiler
- CUDA toolkit
- NVIDIA cuTENSOR / cuQuantum components as required by your PETSc / backend setup
 
A GPU is not required for basic code inspection, but it is required for the main optimized execution path described here.
 
## Build Instructions
 
Typical local build flow:
 
```bash
mkdir -p build
cd build
cmake ..
make -j
```
 
This should produce the main executable:
 
```bash
./run_sim
```
 
## Quickstart
 
From the build directory, run the main validation benchmark:
 
```bash
./run_sim -use_gpu_aware_mpi 0
```
 
This executes the current trusted validation flow for the batched GPU path.
 
## Validation Workflow
 
The repository maintains a strict small-system validation fixture to prevent performance optimizations from changing the physics result.
 
Current validation configuration:
 
- transmons: `N=2`
- cutoff dimension: `d=2`
- batched TS steps: `1000`
- batch sizes: `1`, `2`, `4`
 
The validation checks:
 
- State 0 for `B=1` matches the trusted baseline within tolerance
- State 0 for `B=2` matches the trusted baseline within tolerance
- State 0 is identical across `B=1` and `B=2`
- the `(0,0)` element stays in the expected regime
 
Current trusted State 0 baseline:
 
```text
(0.618762,0) (-0.0116998,-0.00646817) (0.0359727,0.0187981) (0.0114843,0.00366108)
(-0.0116998,0.00646817) (0.0373947,0) (-0.0355948,-0.0742133) (0.00937813,0.0117711)
(0.0359727,-0.0187981) (-0.0355948,0.0742133) (0.334491,0) (-0.0370112,0.00588792)
(0.0114843,-0.00366108) (0.00937813,-0.0117711) (-0.0370112,-0.00588792) (0.00935285,0)
```
 
A successful validation run should report:
 
- zero difference between batch-1 and batch-2 State 0
- small numerical difference to the trusted baseline
- `(0,0)` remaining above `0.61`
 
## Stress-Test Mode
 
The CLI also provides a stress-test mode for larger scaling experiments using the same underlying benchmark/evolution path.
 
Example GPU-only qutrit stress test:
 
```bash
./run_sim -use_gpu_aware_mpi 0 \
  -stress_test 1 \
  -stress_run_cpu 0 \
  -stress_num_transmons 5 \
  -stress_cutoff_dim 3 \
  -stress_batched_num_steps 250 \
  -stress_batch_sizes 8,16
```
 
Representative single-case run:
 
```bash
./run_sim -use_gpu_aware_mpi 0 \
  -stress_test 1 \
  -stress_run_cpu 0 \
  -stress_num_transmons 5 \
  -stress_cutoff_dim 3 \
  -stress_batched_num_steps 250 \
  -stress_batch_sizes 16
```
 
Stress mode is intended for profiling, scaling studies, and bottleneck identification.
 
## Reproducing Key Results
 
At the current stage of the project, the main reproducible results are:
 
1. Small-system validation:
 
```bash
./run_sim -use_gpu_aware_mpi 0
```
 
2. Qutrit stress timing:
 
```bash
./run_sim -use_gpu_aware_mpi 0 \
  -stress_test 1 \
  -stress_run_cpu 0 \
  -stress_num_transmons 5 \
  -stress_cutoff_dim 3 \
  -stress_batched_num_steps 250 \
  -stress_batch_sizes 8,16
```
 
3. Representative qutrit profiling:
 
```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --stats=true \
  -o profiling/qutrit_profile \
  ./run_sim -use_gpu_aware_mpi 0 \
  -stress_test 1 \
  -stress_run_cpu 0 \
  -stress_num_transmons 5 \
  -stress_cutoff_dim 3 \
  -stress_batched_num_steps 250 \
  -stress_batch_sizes 16
```
 
Relevant profiling notes are stored in `profiling/`.
 
## Data Provenance
 
All benchmark and validation data in the current repository are generated by running the solver itself. The current workflow does not depend on an external dataset.
 
The main provenance categories are:
 
- validation outputs generated by the trusted `N=2`, `d=2` run
- stress-test outputs generated by the configurable CLI stress mode
- profiling summaries generated from local `nsys` runs and recorded in `profiling/*.md`
 
## Example Workflow
 
A minimal end-to-end workflow for an external reader is:
 
1. Build the project with CMake.
2. Run the default validation command.
3. Confirm that batch-1 and batch-2 State 0 match and that the baseline is reproduced.
4. Run one qutrit stress-test command.
5. Optionally profile the qutrit stress case with `nsys`.
6. Compare the stress behavior with the notes in `profiling/`.
 
## Why Matrix-Free?
 
For a system with local dimension `d` and subsystem count `N`, the Hilbert-space dimension is `D = d^N`, while the density matrix has `D^2` entries. Explicit Liouvillian construction therefore scales extremely poorly.
 
Instead of assembling the full superoperator, cuLindblad evaluates the action of the RHS directly on the density matrix:
 
```text
dρ/dt = -i [H, ρ] + Σ_k (L_k ρ L_k† - 1/2 {L_k†L_k, ρ})
```
 
This shifts the problem from explicit superoperator storage toward local grouped tensor contractions and structured local operator application.
 
## Current Limitations
 
The current repository is still an active prototype.
 
Current limitations include:
 
- the strongest performance story is still for single-node GPU execution
- the full-stack export path to reduced channels / Stim is not yet complete
- the current optimization story is benchmarked most strongly on the validated transmon fixture and qutrit stress cases
- multi-node and multi-GPU scaling are future directions rather than mature final features
 
## Future Directions
 
Planned next steps include:
 
- generating reduced process matrices / effective channels for downstream tools
- integrating those artifacts into higher-level simulators such as Stim
- extending scaling beyond single-node GPU execution
- developing multi-node / multi-GPU support
- improving Python-facing usability and backend integration
- expanding structured fast paths where mathematically justified
 
## Validation Philosophy
 
A central principle of this repository is that no optimization should be accepted purely on runtime. Optimizations are accepted only when:
 
- the trusted validation fixture still passes
- batch consistency is preserved
- the physical result remains stable within tolerance
 
This is why the repository includes both a strict validation mode and a larger stress-test mode.
 
## References and Dependencies
 
Core external software components include:
 
- PETSc
- CUDA
- NVIDIA cuTENSOR / cuQuantum components
- standard C++17 tooling
 
The scientific context also builds on standard open quantum systems and numerical simulation references, including QuTiP for small-system validation comparisons.
 
## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
