# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 16:11:42 2025

@author: SCD UM
"""

# -*- coding: utf-8 -*-
"""
Analyse du comportement de l'optimisation ADRS en fonction du raffinement du maillage
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
def ADRS(NX, xcontrol, Cible, adapt_mesh=False):
    K = 0.1
    L = 1.0
    Temps = 20.0
    V = 1.0
    lamda = 1.0

    NT = 1000
    eps = 0.001

    dx = L / (NX - 1)
    dt = 0.5 * dx**2 / (V * dx + 2 * K)

    x = np.linspace(0.0, 1.0, NX)
    T = np.zeros(NX)
    F = np.zeros(NX)
    RHS = np.zeros(NX)

    # Terme source
    for j in range(1, NX - 1):
        for ic in range(len(xcontrol)):
            F[j] += xcontrol[ic] * np.exp(-100 * (x[j] - L / (ic + 1))**2)

    # Ajustement du pas de temps
    dt = 0.5 * dx**2 / (V * dx + 2 * K + abs(np.max(F)) * dx**2)

    n = 0
    res = 1
    res0 = 1

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
        if n == 1:
            res0 = res
            
    coût = np.dot(T - Cible, T - Cible) * dx
    return coût, T, x


# ================================================================
#  Interpolation entre maillages
# ================================================================
def interpolate_to_common_mesh(x_old, T_old, x_new):
    interp_func = interp1d(x_old, T_old, kind='linear', fill_value="extrapolate")
    return interp_func(x_new)


# ================================================================
#  Calcul des matrices A et B (produits scalaires L2)
# ================================================================
def compute_A_B(NX, nbc, xcible, Cible, adapt_mesh=False):
    A = np.zeros((nbc, nbc))
    B = np.zeros(nbc)
    xcontrol = np.zeros(nbc)

    coût, T0, x = ADRS(NX, xcontrol, Cible, adapt_mesh)

    for ic in range(nbc):
        xic = np.zeros(nbc)
        xic[ic] = 1
        coût, Tic, x_ic = ADRS(NX, xic, Cible, adapt_mesh)

        if adapt_mesh:
            Tic = interpolate_to_common_mesh(x_ic, Tic, x)

        B[ic] = np.dot((Cible - T0), Tic) / (NX - 1)

        for jc in range(0, ic + 1):
            xjc = np.zeros(nbc)
            xjc[jc] = 1
            coût, Tjc, x_jc = ADRS(NX, xjc, Cible, adapt_mesh)

            if adapt_mesh:
                Tjc = interpolate_to_common_mesh(x_jc, Tjc, x)

            A[ic, jc] = np.dot(Tic, Tjc) / (NX - 1)

    for ic in range(nbc):
        for jc in range(ic, nbc):
            A[ic, jc] = A[jc, ic]

    return A, B, x


# ================================================================
#  Fonctionnelle à minimiser (appelée par scipy)
# ================================================================
def fonctionnel(x, NX, nbc, xcible, Cible, adapt_mesh=False):
    coût, T, _ = ADRS(NX, x, Cible, adapt_mesh)
    return coût


# ================================================================
#  BOUCLE SUR LES TAILLES DE MAILLAGE
# ================================================================
nbc = 4
xcible = np.array([1.0, 2.0, 3.0, 4.0])

NX_values = range(4, 11)  # NX = 4 à 10
plt.figure(figsize=(12, 6))

for NX in NX_values:
    print("\n===============================")
    print(f"Calcul pour NX = {NX}")
    print("===============================")

    coût, Cible, x = ADRS(NX, xcible, np.zeros(NX), adapt_mesh=False)
    A, B, x_common = compute_A_B(NX, nbc, xcible, Cible, adapt_mesh=True)

    xopt = np.linalg.solve(A, B)
    coût_opt_lin, T_opt_lin, _ = ADRS(NX, xopt, Cible, adapt_mesh=True)

    # Tracé sur le même graphique
    plt.plot(x_common, T_opt_lin, label=f"NX={NX}")

plt.plot(x_common, Cible, 'k--', linewidth=2, label="Cible (référence)")
plt.xlabel("x (domaine)")
plt.ylabel("Solution T(x)")
plt.title("Comparaison des solutions optimales pour différents maillages NX")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
