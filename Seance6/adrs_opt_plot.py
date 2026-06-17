# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 12:23:54 2025

@author: SCD UM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =======================================================
# Fonctions utilitaires
# =======================================================

def solve_ADRS(NX, xcontrol):
    """
    Résout une version simplifiée de l'équation ADRS :
    u_t = -V u_x + K u_xx - λ u + f(x)
    en stationnaire (pour illustrer l'optimisation).
    """
    V, K, lam = 1.0, 0.1, 1.0
    L = 1.0
    dx = L / (NX - 1)
    x = np.linspace(0, L, NX)
    u = np.zeros(NX)
    F = np.zeros(NX)

    # source = somme des gaussiennes centrées sur chaque point de contrôle
    for ic in range(len(xcontrol)):
        F += xcontrol[ic] * np.exp(-100 * (x - (ic + 1) / (len(xcontrol) + 1))**2)

    # schéma stationnaire : -V u_x + K u_xx - λ u + F = 0
    A = np.zeros((NX, NX))
    b = F.copy()
    for i in range(1, NX - 1):
        A[i, i - 1] = -K / dx**2 - V / (2 * dx)
        A[i, i] = 2 * K / dx**2 + lam
        A[i, i + 1] = -K / dx**2 + V / (2 * dx)
    # conditions de bord : u(0)=u(L)=0
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    b[0] = 0.0
    b[-1] = 0.0

    u = np.linalg.solve(A, b)
    return x, u


# =======================================================
# Construction du problème inverse
# =======================================================

def compute_matrices(NX, nbc, u_des):
    """Construit les matrices A et B selon la linéarité ADRS"""
    xcontrol0 = np.zeros(nbc)
    x, u0 = solve_ADRS(NX, xcontrol0)

    U = []
    for ic in range(nbc):
        e = np.zeros(nbc)
        e[ic] = 1.0
        _, ui = solve_ADRS(NX, e)
        U.append(ui)

    U = np.array(U)
    A = np.zeros((nbc, nbc))
    B = np.zeros(nbc)

    for i in range(nbc):
        for j in range(nbc):
            A[i, j] = np.trapz(U[i] * U[j], x)
        B[i] = np.trapz(U[i] * (u_des - u0), x)

    return A, B, U, u0, x


# =======================================================
# Paramètres du problème
# =======================================================
nbc = 4
X_opt_ref = np.array([1.0, 2.0, 3.0, 4.0])

# Boucle de raffinement
NX_values = [30, 40, 60, 80, 120]
X_results = []
J_results = []

for NX in NX_values:
    x, u_des = solve_ADRS(NX, X_opt_ref)  # solution cible
    A, B, U, u0, xgrid = compute_matrices(NX, nbc, u_des)
    X_star = np.linalg.solve(A, B)
    u_rec = u0 + np.dot(X_star, U)
    J_val = 0.5 * np.trapz((u_rec - u_des)**2, xgrid)

    X_results.append(X_star)
    J_results.append(J_val)

# Conversion en tableau pour tracés
X_results = np.array(X_results)
errors = np.linalg.norm(X_results - X_opt_ref, axis=1)


# =======================================================
# 🔹 Figures pour le rapport
# =======================================================

# 1️⃣ Convergence des composantes de X*(h)
plt.figure(figsize=(8,5))
for i in range(nbc):
    plt.plot(1/np.array(NX_values), X_results[:, i], 'o--', label=f"$x_{i+1}^*$")
    plt.axhline(X_opt_ref[i], linestyle=':', color='gray')
plt.xlabel("Pas h = 1/(N_X-1)")
plt.ylabel("Composantes de X*(h)")
plt.title("Convergence des composantes du contrôle optimal")
plt.legend()
plt.grid(True)
plt.savefig("Xstar_convergence.png", dpi=300)
plt.show()

# 2️⃣ Fonctionnelle de coût et erreur
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(1/np.array(NX_values), J_results, 'o--')
plt.xlabel("h")
plt.ylabel("J(X*(h))")
plt.title("Évolution de la fonctionnelle de coût")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(1/np.array(NX_values), errors, 's--', color='r')
plt.xlabel("h")
plt.ylabel("||X*(h) - Xopt||")
plt.title("Erreur sur le contrôle optimal")
plt.grid(True)
plt.tight_layout()
plt.savefig("J_and_Error_vs_h.png", dpi=300)
plt.show()

# 3️⃣ Cas cible u_des = 1 (uniforme)
NX = 80
u_des = np.ones(NX)
A, B, U, u0, x = compute_matrices(NX, nbc, u_des)
X_star = np.linalg.solve(A, B)

plt.figure(figsize=(8,5))
plt.bar(range(1, nbc+1), X_star, color='orange')
plt.xlabel("Indice du contrôle j")
plt.ylabel("x*_j obtenu")
plt.title("Composantes de X* pour u_des = 1")
plt.savefig("Xstar_u_des_1.png", dpi=300)
plt.show()

# 4️⃣ Surface du coût J(x1, x2)
NX = 60
x, u_des = solve_ADRS(NX, X_opt_ref)
A, B, U, u0, _ = compute_matrices(NX, nbc, u_des)
x1_range = np.linspace(0, 3, 40)
x2_range = np.linspace(0, 3, 40)
Jsurf = np.zeros((len(x1_range), len(x2_range)))

for i, x1 in enumerate(x1_range):
    for j, x2 in enumerate(x2_range):
        Xtmp = np.array([x1, x2, X_opt_ref[2], X_opt_ref[3]])
        u_rec = u0 + np.dot(Xtmp, U)
        Jsurf[i,j] = 0.5 * np.trapz((u_rec - u_des)**2, x)

X1, X2 = np.meshgrid(x1_range, x2_range)
fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, Jsurf.T, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('J(x1, x2)')
ax.set_title('Surface de la fonctionnelle J(x1, x2)')
plt.savefig("Cost_surface_X1X2.png", dpi=300)
plt.show()
