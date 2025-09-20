import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. Initialisation
K, L, V, lambda_, eps = 0.1, 1.0, 1.0, 1.0, 0.001
niter_refinement = 10
NX = 100

# 2. Boucle de raffinement
h_values = []
norm_L2 = []
norm_H1 = []
semi_H2 = []

for _ in range(niter_refinement):
    NX += 5
    h = L / (NX - 1)
    h_values.append(h)
    x = np.linspace(0, L, NX)

    # Solution exacte
    u_ex = np.exp(-20 * (x - 0.5)**2)
    u_ex_prime = -40 * (x - 0.5) * u_ex
    u_ex_second = (-20 + 1600 * (x - 0.5)**2) * u_ex

    # Résolution numérique (simplifiée)
    T = np.zeros(NX)  # À remplacer par votre solution numérique

    # Calcul des normes
    diff = u_ex - T
    norm_L2.append(np.sqrt(np.sum(diff[1:-1]**2) * h))
    norm_H1.append(np.sqrt(norm_L2[-1]**2 + np.sum(((u_ex_prime[2:] - u_ex_prime[:-2]) / (2*h) - (T[2:] - T[:-2]) / (2*h))**2) * h))
    semi_H2.append(np.sqrt(np.sum(u_ex_second[1:-1]**2) * h))

# 6. Identification de C et k
log_h = np.log(h_values)
log_norm_L2 = np.log(norm_L2)

def model(h, log_C, k):
    return log_C + k * h

popt, _ = curve_fit(model, log_h, log_norm_L2, p0=[0, 2])
log_C_opt, k_opt = popt
C_opt = np.exp(log_C_opt)

# 7. Tracé des courbes
plt.figure()
plt.loglog(h_values, norm_L2, 'o-', label='Erreur L2')
plt.loglog(h_values, [C_opt * h**k_opt for h in h_values], '--', label=f'$C h^{k_opt:.2f}$')
plt.xlabel('Taille du maillage $h$')
plt.ylabel('Erreur $L^2$')
plt.legend()
plt.grid(True)
plt.show()
