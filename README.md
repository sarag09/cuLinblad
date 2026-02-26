# cpp-lindblad-propagator

## 1. Elevator Pitch
"I propose to develop a high-performance C++ library for solving the Lindblad master equation to output the system propagator for open quantum systems. This tool will feature a hybrid parallelization scheme, supporting both MPI-based distributed computing and multi-node multi-GPU acceleration to significantly outperform standard single-node solvers."

## 2. Project Partner
"I am currently solo."

## 3. 
- **Project form:** Tool/library (High-performance C++ solver).
- **Baseline/comparison:** Compare performance against QuTiP, single-threaded C++, and internal CPU vs. GPU benchmarks.
- **Full-stack lever:** Access to larger Hilbert spaces for open-system dynamics, enabling the characterization of larger NISQ devices than currently possible with standard tools.

## 4. Motivating Reference
J.R. Johansson, P.D. Nation, and F. Nori, "QuTiP: An open-source Python framework for the dynamics of open quantum systems.", Comp. Phys. Comm. 183, 1760–1772 (2012).
