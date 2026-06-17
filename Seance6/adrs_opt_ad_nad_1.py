# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 16:38:28 2025

@author: SCD UM
"""

# -*- coding: utf-8 -*-
"""
Comparaison entre solution admissible (Neumann) et non admissible
Auteur : SCD UM
Date : Oct 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d


# ================================================================
#  FONCTION ADRS : résolution de l’équation advection-diffusion-réaction-source
# ================================================================
def ADRS(NX, xcontrol, Cible, neumann=True):
    K = 0.1
    L = 1.0
    V = 1.0
    lamda = 1.0
    NT = 1000
    eps = 0.001

    dx = L / (NX - 1)
    dt = 0.5 * dx**2 / (V * dx + 2 * K)

    x = np.linspace(0.0, L, NX)
    T = np.zeros(NX)
    F = np.zeros(NX)
    RHS = np.zeros(NX)

    # === Terme source ===
    for j in range(1, NX - 1):
        for ic in range(len(xcontrol)):
            F[j] += xcontrol[ic] * np.exp(-100 * (x[j] - L / (ic + 1))**2)

    # Ajustement du pas de temps
    dt = 0.5 * dx**2 / (V * dx + 2 * K + abs(np.max(F)) * dx**2)

    n = 0
    res = 1
    res0 = 1

    # === Boucle temporelle ===
    while n < NT and res > eps * res0:
        n += 1
        res = 0
        for j in range(1, NX - 1):
            xnu = K + 0.5 * dx * abs(V)
            Tx = (T[j + 1] - T[j - 1]) / (2 * dx)
            Txx = (T[j - 1] - 2 * T[j] + T[j + 1]) / (dx**2)
            RHS[j] = dt * (-V * Tx + xnu * Txx - lamda * T[j] + F[j])
            res += abs(RHS[j])

        for j in range(1, NX - 1):
            T[j] += RHS[j]

        # === Conditions aux limites ===
        T[0] = 0.0  # Dirichlet à gauche
        if neumann:
            # Condition de Neumann : ∂T/∂x = 0 → extrapolation
            T[-1] = 2 * T[-2] - T[-3]
        else:
            # Cas non admissible : on impose une perturbation artificielle
            T[-1] = T[-2] + 0.3 * np.sin(10 * T[-2])
            #T[-1] = 2 * T[-2] - T[-3]

        if n == 1:
            res0 = res

    coût = np.dot(T - Cible, T - Cible) * dx
    return coût, T, x


# ================================================================
#  Programme principal
# ================================================================
nbc = 4
NX = 50
xcible = np.array([1.0, 2.0, 3.0, 4.0])



# Fonction cible : sinusoïde
x_ref = np.linspace(0, 1, NX)
Cible = np.sin(np.pi * x_ref)   
#Cible = np.sin(2*pi*x_ref)
#Cible = np.exp(-100 * (x_ref - 0.5)**2)

# === Calcul de la solution admissible (Neumann) ===
coût_adm, T_adm, x = ADRS(NX, xcible, Cible, neumann=True)

# === Calcul de la solution non admissible (violation de Neumann) ===
coût_nonadm, T_nonadm, _ = ADRS(NX, xcible, Cible, neumann=False)
#NX_values = range(4, 11)
# ================================================================
#  Tracé des résultats
# ================================================================
plt.figure(figsize=(10, 6))
plt.plot(x, T_adm, 'b-', linewidth=2, label="Solution admissible (Neumann)")
plt.scatter(x, T_nonadm, color='r', label="Solution non admissible (nuage)")
plt.plot(x, Cible, 'k--', linewidth=2, label="Cible (sinus)")

plt.xlabel("x (domaine)")
plt.ylabel("T(x)")
plt.title("Comparaison entre solution admissible et non admissible (condition de Neumann)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================================================
# Vérification numérique de la condition de Neumann
# ================================================================
dTdx = (T_adm[-1] - T_adm[-2]) / (x[-1] - x[-2])
print(f"Dérivée ∂T/∂x (bord droit, admissible) = {dTdx:.3e}")
