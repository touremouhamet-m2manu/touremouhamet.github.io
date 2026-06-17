# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 10:36:47 2025

@author: SCD UM
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def ADRS(NX, xcontrol, Cible):
    # u,t = -V u,x + k u,xx - lambda u + f

    # Paramètres physiques
    K = 0.1  # Coefficient de diffusion
    L = 1.0  # Taille du domaine
    Temps = 20.0  # Temps d'intégration

    V = 1.0
    lamda = 1.0

    # Paramètres numériques
    NT = 1000  # Nombre de pas de temps max
    ifre = 1000000  # Tracer toutes les itérations de temps ifre
    eps = 0.0001  # Rapport de convergence relative

    dx = L / (NX - 1)  # Pas de grille (espace)
    dt = dx**2 / (V * dx + K + dx**2)  # Condition de pas de grille (temps) CFL de stabilité

    # Initialisation
    x = np.linspace(0.0, 1.0, NX)
    T = np.zeros(NX)
    F = np.zeros(NX)
    reste = []
    RHS = np.zeros(NX)

    for j in range(1, NX - 1):
        for ic in range(len(xcontrol)):
            F[j] += xcontrol[ic] * np.exp(-100 * (x[j] - L / (ic + 1))**2)

    dt = 0.5 * dx**2 / (V * dx + 2 * K + abs(np.max(F)) * dx**2)  # Recalcul du pas de temps

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
            RHS[j] = 0

        if n == 1:
            res0 = res

        reste.append(res)

    coût = np.dot(T - Cible, T - Cible) * dx  # Intégrale de Riemann de J
    return coût, T

# Paramètres pour l'optimisation
nbc = 6
NX = 30
nb_iter_refine = 1
meilleur_coût = 1.e10
x_best = np.zeros(nbc)
cost_tab = np.zeros(nb_iter_refine)
NX_tab = np.zeros(nb_iter_refine)

for irefine in range(nb_iter_refine):
    NX += 5
    NX_tab[irefine] = NX
    Cible = np.zeros(NX)
    xcible = np.arange(nbc) + 1
    coût_indésirable, Cible = ADRS(NX, xcible, Cible)

    xcontrol = np.zeros(nbc)
    coût, T0 = ADRS(NX, xcontrol, Cible)

    A = np.zeros((nbc, nbc))
    B = np.zeros(nbc)

    for ic in range(nbc):
        xic = np.zeros(nbc)
        xic[ic] = 1
        coût, Tic = ADRS(NX, xic, Cible)
        B[ic] = np.dot((Cible - T0), Tic) / (NX - 1)
        for jc in range(0, ic + 1):
            xjc = np.zeros(nbc)
            xjc[jc] = 1
            coût, Tjc = ADRS(NX, xjc, Cible)
            A[ic, jc] = np.dot(Tic, Tjc) / (NX - 1)

    for ic in range(nbc):
        for jc in range(ic, nbc):
            A[ic, jc] = A[jc, ic]

    xopt = np.linalg.solve(A, B)
    print("Xopt =", xopt)
    coût_opt, T = ADRS(NX, xopt, Cible)
    print("Coût optimal =", coût_opt)
    cost_tab[irefine] = coût_opt

    if meilleur_coût >= coût_opt:
        meilleur_coût = coût_opt
        T_opt = T.copy()
        x_best = xopt.copy()
        Target_opt = Cible.copy()

# Visualisation des résultats
plt.plot(NX_tab, np.log10(cost_tab))
plt.xlabel("Taille de la maille")
plt.ylabel("Log10(Coût)")
plt.show()

plt.plot(T_opt, label="Optimisation linéaire")
plt.plot(Target_opt, label="Cible")
plt.xlabel("Domaine x")
plt.ylabel("Solution T")
plt.legend()
plt.show()

# Utilisation de l'optimiseur Python
def fonctionnel(x):
    NX = 28
    Cible = np.zeros(NX)
    xcible = np.arange(nbc) + 1
    coût, cible = ADRS(NX, xcible, Cible)
    coût, T = ADRS(NX, x, cible)
    return coût

x0 = np.zeros(nbc)
options = {"maxiter": 100, 'disp': True}
res = minimize(fonctionnel, x0, options=options)
print("------------------------------------------------")
print(res)
print("------------------------------------------------")
