import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# PARAMÈTRES PHYSIQUES
K = 0.1
L = 1.0
V = 1.0
lambda_ = 1.0
eps = 0.001
niter_refinement = 10
NX = 10  # Nombre initial de points

# Solution exacte et ses dérivées
def u_ex(x):
    return np.exp(-20 * (x - 0.5)**2)

def u_ex_prime(x):
    return -40 * (x - 0.5) * u_ex(x)

def u_ex_second(x):
    return (-20 + 1600 * (x - 0.5)**2) * u_ex(x)

# Initialisation des tableaux
norm_L2 = np.zeros(niter_refinement)
norm_H1 = np.zeros(niter_refinement)
semi_H2 = np.zeros(niter_refinement)
h_values = np.zeros(niter_refinement)

for iter in range(niter_refinement):
    NX += 5
    dx = L / (NX - 1)
    h_values[iter] = dx
    x = np.linspace(0.0, L, NX)

    # Solution exacte discrétisée
    u_exact = u_ex(x)
    u_exact_prime = u_ex_prime(x)
    u_exact_second = u_ex_second(x)

    # Initialisation de la solution numérique (à remplir)
    T = np.zeros(NX)

    # Résolution numérique (votre code ici)
    # ...

    # Calcul des différences
    diff = u_exact - T
    diff_prime = u_exact_prime.copy()
    diff_prime[1:-1] = (u_exact[2:] - u_exact[:-2]) / (2 * dx) - (T[2:] - T[:-2]) / (2 * dx)

    # Normes
    norm_L2[iter] = np.sqrt(np.sum(diff[1:-1]**2) * dx)
    norm_H1[iter] = np.sqrt(norm_L2[iter]**2 + np.sum(diff_prime[1:-1]**2) * dx)
    semi_H2[iter] = np.sqrt(np.sum(u_exact_second[1:-1]**2) * dx)

# Filtrer les valeurs valides
valid_indices = (h_values > 0) & (norm_L2 > 0)
h_values_valid = h_values[valid_indices]
norm_L2_valid = norm_L2[valid_indices]

# Prendre le logarithme
log_h = np.log(h_values_valid)
log_norm_L2 = np.log(norm_L2_valid)

# Définir le modèle
def model(h, log_C, k):
    return log_C + k * h

# Valeurs initiales pour l'optimisation
initial_guess = [np.log(1.0), 2.0]

# Optimisation
try:
    popt, pcov = curve_fit(model, log_h, log_norm_L2, p0=initial_guess)
    log_C_opt, k_opt = popt
    C_opt = np.exp(log_C_opt)
    print(f"Paramètres optimaux : C = {C_opt:.4f}, k = {k_opt:.4f}")
except RuntimeError as e:
    print(f"Erreur lors de l'optimisation : {e}")

# Tracé des courbes
plt.figure(figsize=(10, 6))
plt.loglog(h_values_valid, norm_L2_valid / semi_H2[valid_indices], 'o-', label=r'$|u - u_h|_{L^2} / |u|_{H^2}$')
h_range = np.logspace(np.log10(h_values_valid[-1]), np.log10(h_values_valid[0]), 100)
plt.loglog(h_range, C_opt * h_range**k_opt, '--', label=f'$C h^{k_opt:.2f}$')
plt.loglog(h_range, C_opt * h_range**(k_opt + 1), '--', label=f'$C h^{k_opt + 1:.2f}$')
plt.xlabel('Taille du maillage $h$')
plt.ylabel('Erreur normalisée')
plt.title('Convergence en espace')
plt.legend()
plt.grid(True)
plt.show()
