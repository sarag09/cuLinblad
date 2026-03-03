import sys
import os

# ignore gpu aware error mrssage
os.environ["PETSC_OPTIONS"] = "-use_gpu_aware_mpi 0"

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# 1. Tell Python where to find your compiled C++ library
sys.path.append(os.path.abspath("build"))
import cuLindblad_core

print("Starting Check-In #1 Validation...")

# --- Part 1: The QuTiP Ground Truth ---
print("Running QuTiP Simulation...")
coupling = 2.0
gamma = 0.5

H = coupling * (qt.tensor(qt.sigmax(), qt.sigmax()) + qt.tensor(qt.sigmay(), qt.sigmay()))
c_ops = [np.sqrt(gamma) * qt.tensor(qt.destroy(2), qt.qeye(2))]
rho0 = qt.ket2dm(qt.tensor(qt.basis(2, 1), qt.basis(2, 0))) # |10> state

tlist = np.linspace(0, 1.0, 100) # 100 time steps for a smooth plot
result = qt.mesolve(H, rho0, tlist, c_ops, [])
qutip_pop_00 = [rho.full()[0, 0].real for rho in result.states]

# --- Part 2: The cuLindblad C++ Engine ---
print("Running C++ cuLindblad Engine...")
cpp_final_result = cuLindblad_core.run_simulation(coupling, gamma)

# --- Part 3: The Validation Plot ---
print("Generating Plot...")
plt.figure(figsize=(8, 5))

# Plot QuTiP as a smooth line
plt.plot(tlist, qutip_pop_00, label="QuTiP (Dense Matrix)", color='blue', linewidth=2)

# Plot our C++ result as a single Red Star at t=1.0
plt.plot(1.0, cpp_final_result, 'r*', markersize=15, label="cuLindblad (Matrix-Free C++)")

plt.title("Check-In #1: 2-Qubit Matrix-Free Validation")
plt.xlabel("Time")
plt.ylabel("Population of |00> state")
plt.legend()
plt.grid(True)

# Save the plot so you can put it in your Memo PDF!
plt.savefig("check_in_1_validation.png", dpi=300)
print("Success! Plot saved as 'check_in_1_validation.png'")