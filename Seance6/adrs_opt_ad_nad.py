# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 16:28:53 2025

@author: SCD UM
"""

import numpy as np
import matplotlib.pyplot as plt

def ADRS(NX, admissible=True):
    """
    Résout l'équation ADRS 1D dans [0,1]
    - admissible=True : impose condition de Neumann correcte (dT/dx=0 au bord)
    - admissible=False : viole la condition de Neumann
    """
    # Paramètres physiques
    K = 0.1
    L = 1.0
    V = 1.0
    lamda = 1.0

    # Maillage
    x = np.linspace(0, 1, NX)
    dx = x[1] - x[0]

    # Condition cible (fonction sinus)
    Cible = np.sin(2 * np.pi * x)

    # Initialisation
    T = np.zeros_like(x)
    NT = 300
    dt = 0.5 * dx**2 / (V * dx + 2 * K)
    
    # Terme source pour stabiliser la solution
    F = np.exp(-100 * (x - 0.5)**2)

    for n in range(NT):
        Tnew = T.copy()
        for j in range(1, NX-1):
            xnu = K + 0.5 * dx * abs(V)
            Tx = (T[j+1] - T[j-1]) / (2*dx)
            Txx = (T[j-1] - 2*T[j] + T[j+1]) / (dx**2)
            Tnew[j] = T[j] + dt * (-V * Tx + xnu * Txx - lamda * T[j] + F[j])

        # Condition de Neumann au bord (∂T/∂x = 0)
        if admissible:
            # Condition correcte
            Tnew[0] = Tnew[1]
            Tnew[-1] = Tnew[-2]
        else:
            # Condition erronée (non admissible)
            Tnew[0] = 2*Tnew[1] - 3*Tnew[2]   # Erreur volontaire
            Tnew[-1] = -Tnew[-2]              # Bord instable

        T = Tnew.copy()

    return x, T, Cible


# ============================================================
# Calcul des deux cas : admissible et non admissible
# ============================================================

x1, T_adm, Cible = ADRS(NX=50, admissible=True)
x2, T_non, _ = ADRS(NX=50, admissible=False)

# ============================================================
# Tracé
# ============================================================

plt.figure(figsize=(10,5))
plt.plot(x1, T_adm, 'b-', linewidth=2, label="✅ Solution admissible (Neumann)")
plt.scatter(x2, T_non, color='red', s=30, label="❌ Solution non admissible (instable)")
plt.plot(x1, Cible, 'k--', label="Cible : sin(2πx)")

plt.xlabel("x", fontsize=12)
plt.ylabel("T(x)", fontsize=12)
plt.title("Comparaison : solution admissible vs non admissible (condition de Neumann)")
plt.legend()
plt.grid(True)
plt.show()
