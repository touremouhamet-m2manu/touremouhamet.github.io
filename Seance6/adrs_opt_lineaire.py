# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 10:46:51 2025

@author: SCD UM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def ADRS(NX, xcontrol, Cible, adapt_mesh=False):
    # Paramètres physiques
    K = 0.1  # Coefficient de diffusion
    L = 1.0  # Taille du domaine
    Temps = 20.0  # Temps d'intégration
    V = 1.0  # Vitesse d'advection
    lamda = 1.0  # Coefficient de réaction

    # Paramètres numériques
    NT = 1000  # Nombre de pas de temps max
    eps = 0.0001  # Critère de convergence

    dx = L / (NX - 1)  # Pas d'espace
    dt = 0.5 * dx**2 / (V * dx + 2 * K)  # Pas de temps initial

    # Initialisation
    x = np.linspace(0.0, 1.0, NX)
    T = np.zeros(NX)
    F = np.zeros(NX)
    RHS = np.zeros(NX)

    # Calcul du terme source F
    for j in range(1, NX - 1):
        for ic in range(len(xcontrol)):
            F[j] += xcontrol[ic] * np.exp(-100 * (x[j] - L / (ic + 1))**2)

    # Recalcul du pas de temps en fonction de F
    dt = 0.5 * dx**2 / (V * dx + 2 * K + abs(np.max(F)) * dx**2)

    # Boucle principale en temps
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

    # Calcul du coût
    coût = np.dot(T - Cible, T - Cible) * dx
    return coût, T, x

def interpolate_to_common_mesh(x_old, T_old, x_new):
    # Interpolation linéaire pour adapter les solutions à un maillage commun
    interp_func = interp1d(x_old, T_old, kind='linear', fill_value="extrapolate")
    T_new = interp_func(x_new)
    return T_new

def compute_A_B(NX, nbc, xcible, Cible, adapt_mesh=False):
    # Initialisation des matrices A et B
    A = np.zeros((nbc, nbc))
    B = np.zeros(nbc)

    # Calcul de la solution initiale T0
    xcontrol = np.zeros(nbc)
    coût, T0, x = ADRS(NX, xcontrol, Cible, adapt_mesh)

    # Calcul des solutions élémentaires Tic et Tjc
    for ic in range(nbc):
        xic = np.zeros(nbc)
        xic[ic] = 1
        coût, Tic, x_ic = ADRS(NX, xic, Cible, adapt_mesh)

        # Interpolation si adaptation de maillage
        if adapt_mesh:
            Tic = interpolate_to_common_mesh(x_ic, Tic, x)

        B[ic] = np.dot((Cible - T0), Tic) / (NX - 1)

        for jc in range(0, ic + 1):
            xjc = np.zeros(nbc)
            xjc[jc] = 1
            coût, Tjc, x_jc = ADRS(NX, xjc, Cible, adapt_mesh)

            # Interpolation si adaptation de maillage
            if adapt_mesh:
                Tjc = interpolate_to_common_mesh(x_jc, Tjc, x)

            A[ic, jc] = np.dot(Tic, Tjc) / (NX - 1)

    # Symétrisation de A
    for ic in range(nbc):
        for jc in range(ic, nbc):
            A[ic, jc] = A[jc, ic]

    return A, B, x

def fonctionnel(x, NX, nbc, xcible, Cible, adapt_mesh=False):
    coût, T, _ = ADRS(NX, x, Cible, adapt_mesh)
    return coût

# Paramètres
nbc = 4
NX = 30
xcible = np.array([1.0, 2.0, 3.0, 4.0])

# Calcul de la solution cible
coût, Cible, x = ADRS(NX, xcible, np.zeros(NX), adapt_mesh=False)

# Calcul des matrices A et B
A, B, x_common = compute_A_B(NX, nbc, xcible, Cible, adapt_mesh=True)

# Résolution du système linéaire
xopt = np.linalg.solve(A, B)
print("Contrôle optimal (linéaire) :", xopt)

# Optimisation avec scipy.optimize.minimize
x0 = np.zeros(nbc)
res = minimize(fonctionnel, x0, args=(NX, nbc, xcible, Cible, True), options={"maxiter": 100, 'disp': True})
print("Contrôle optimal (minimiseur) :", res.x)

# Comparaison des résultats
coût_opt_lin, T_opt_lin, _ = ADRS(NX, xopt, Cible, adapt_mesh=True)
coût_opt_min, T_opt_min, _ = ADRS(NX, res.x, Cible, adapt_mesh=True)

plt.figure(figsize=(12, 6))
plt.plot(x_common, T_opt_lin, label="Optimisation linéaire")
plt.plot(x_common, T_opt_min, label="Optimisation avec minimiseur")
plt.plot(x_common, Cible, label="Cible")
plt.xlabel("Domaine x")
plt.ylabel("Solution T")
plt.legend()
plt.title("Comparaison des solutions optimisées")
plt.show()
