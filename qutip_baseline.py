import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the 2-Qubit Physics Parameters
coupling_strength = 2.0
decay_rate = 0.5

# 2. Build the Hamiltonian: H = coupling * (XX + YY)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()
iden = qt.qeye(2)

# Tensor products for 2 qubits
XX = qt.tensor(sx, sx)
YY = qt.tensor(sy, sy)
H = coupling_strength * (XX + YY)

# 3. Build the Collapse Operator (Decay on Qubit 0)
sm = qt.destroy(2) # Lowering operator
c_ops = [np.sqrt(decay_rate) * qt.tensor(sm, iden)]

# 4. Set Initial State (Qubit 0 in |1>, Qubit 1 in |0>)
psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))
rho0 = qt.ket2dm(psi0)

# 5. Time Evolution
tlist = np.linspace(0, 1.0, 11) # Simulate from t=0 to t=1.0 in 10 steps
result = qt.mesolve(H, rho0, tlist, c_ops, [])

# 6. Extract the 00-element (Population of |00>) for all time steps
pop_00 = [rho.full()[0, 0].real for rho in result.states]

# 7. Print the results to the screen
print("Time\t QuTiP rho_00")
for t, p in zip(tlist, pop_00):
    print(f"{t:.1f}\t {p:.6f}")