num_levels = 30

a = qt.destroy(num_levels)

omega_r = 7.0 * 2 * np.pi  # frequency in GHz
kappa = 0.001  # decay rate

omega_d = 7.0 * 2 * np.pi
Omega = 0.0002 * 2 * np.pi # Rabi frequency

H = (omega_d - omega_r) * a.dag() * a  + Omega*(a + a.dag())# Hamiltonian

times = np.linspace(0, 8000, 1001)  # time points

psi0 = qt.basis(num_levels, 0)  # initial state

c_ops = [np.sqrt(kappa) * a]  # collapse operators
e_ops = [a.dag() * a, a, a*a]

result = qt.mesolve(H, psi0, times, c_ops, e_ops)