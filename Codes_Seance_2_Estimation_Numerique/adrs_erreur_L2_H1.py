import numpy as np
import matplotlib.pyplot as plt

# PARAMÈTRES PHYSIQUES
K = 0.01
L = 1.0
V = 1.0
lambda_ = 1.0
eps = 0.001

# Solution exacte
def u_ex(s):
    return np.exp(-10 * (s - L/2)**2)

def f_h(x):
    u = u_ex(x)
    us = -20 * (x - L/2) * u
    uss = (-20 + 400 * (x - L/2)**2) * u
    return V * us - K * uss + lambda_ * u

def calculer_normes(T, x, u_exact):
    h = x[1] - x[0]
    erreur = T - u_exact(x)
    norme_L2 = np.sqrt(np.sum(erreur**2) * h)

    deriv_erreur = np.zeros_like(erreur)
    deriv_erreur[1:-1] = (erreur[2:] - erreur[:-2]) / (2 * h)
    norme_H1 = np.sqrt(norme_L2**2 + np.sum(deriv_erreur**2) * h)

    return norme_L2, norme_H1

def resolution_numerique(NX):
    x = np.linspace(0.0, L, NX)
    dx = L / (NX - 1)
    dt = dx**2 / (V * dx + 2 * K + np.max(np.abs(f_h(x))) * dx**2)

    T = u_ex(x)
    F = f_h(x)
    T[-1] = T[-2]  # Condition u_s(L) = 0

    n = 0
    res = 1
    res0 = np.sqrt(np.sum(T**2) * dx)
    reste = []

    while n < 10000 and res / res0 > eps:
        n += 1
        T_old = T.copy()
        RHS = np.zeros(NX)

        for j in range(1, NX - 1):
            if V > 0:
                Tx = (T[j] - T[j-1]) / dx
            else:
                Tx = (T[j+1] - T[j]) / dx
            Txx = (T[j-1] - 2 * T[j] + T[j+1]) / (dx**2)
            RHS[j] = dt * (-V * Tx + K * Txx - lambda_ * T[j] + F[j])

        for j in range(1, NX - 1):
            T[j] += RHS[j]

        T[-1] = T[-2]  # Condition u_s(L) = 0

        res = np.sqrt(np.sum((T - T_old)**2) * dx)
        reste.append(res)

    return T, x, reste

# --- Simulation pour 5 maillages ---
N_points_list = [3, 10, 50, 100, 200]
erreurs_L2 = []
erreurs_H1 = []
h_list = []

for NX in N_points_list:
    T, x, _ = resolution_numerique(NX)
    norme_L2, norme_H1 = calculer_normes(T, x, u_ex)
    erreurs_L2.append(norme_L2)
    erreurs_H1.append(norme_H1)
    h_list.append(L / (NX - 1))

# --- Tracé des erreurs ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.loglog(h_list, erreurs_L2, 'o-', label='Erreur $L^2$')
plt.xlabel('Taille du maillage $h$')
plt.ylabel('Erreur $L^2$')
plt.title('Erreur $L^2$ en fonction de $h$')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.loglog(h_list, erreurs_H1, 'o-', label='Erreur $H^1$', color='orange')
plt.xlabel('Taille du maillage $h$')
plt.ylabel('Erreur $H^1$')
plt.title('Erreur $H^1$ en fonction de $h$')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Tracé de la convergence ---
_, _, reste = resolution_numerique(100)
plt.figure()
plt.semilogy(reste / reste[0])
plt.xlabel('Itération')
plt.ylabel('Résidu normalisé $||u^{n+1} - u^n||_{L^2}/||u^0||_{L^2}$')
plt.title('Convergence vers la solution stationnaire')
plt.grid(True)
plt.show()
