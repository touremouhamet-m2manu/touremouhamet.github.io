import math
import numpy as np
import matplotlib.pyplot as plt

# PARAMÈTRES PHYSIQUES
K = 0.1  # Coefficient de diffusion
L = 1.0  # Taille du domaine
Temps = 20.0  # Temps d'intégration
V = 1.0
lambda_ = 1.0

# PARAMÈTRES NUMÉRIQUES
NX = 10  # Nombre de points de grille initial
NT = 10000  # Nombre de pas de temps max
ifre = 100000  # Tracer toutes les itérations de temps ifre
eps = 0.001  # Rapport de convergence relative
niter_refinement = 10  # Nombre d'itérations de raffinement
errorL2 = np.zeros(niter_refinement)
erreurH1 = np.zeros(niter_refinement)
semiH2 = np.zeros(niter_refinement)
itertab = np.zeros(niter_refinement)

for iter in range(niter_refinement):
    NX += 5
    dx = L / (NX - 1)  # Pas de grille (espace)
    dt = dx**2 / (V * dx + 4 * K + dx**2)  # Condition CFL
    itertab[iter] = dx
    print(f"Iteration {iter}: dx = {dx}, dt = {dt}")

    # Initialisation
    x = np.linspace(0.0, 1.0, NX)
    T = np.zeros(NX)
    F = np.zeros(NX)
    reste = []
    RHS = np.zeros(NX)
    Tex = np.zeros(NX)

    # Solution exacte discrétisée
    for j in range(1, NX - 1):
        Tex[j] = np.exp(-20 * (j * dx - 0.5)**2)

    # Calcul de F (source)
    for j in range(1, NX - 1):
        Tx = (Tex[j + 1] - Tex[j - 1]) / (2 * dx)
        Txx = (Tex[j + 1] - 2 * Tex[j] + Tex[j - 1]) / (dx**2)
        F[j] = V * Tx - K * Txx + lambda_ * Tex[j]

    # Boucle principale en temps
    n = 0
    res = 1
    res0 = 1
    while n < NT and res / res0 > eps:
        n += 1
        res = 0
        for j in range(1, NX - 1):
            xnu = K + 0.5 * dx * abs(V)
            Tx = (T[j + 1] - T[j - 1]) / (2 * dx)
            Txx = (T[j - 1] - 2 * T[j] + T[j + 1]) / (dx**2)
            RHS[j] = dt * (-V * Tx + xnu * Txx - lambda_ * T[j] + F[j])
            res += abs(RHS[j])

        for j in range(1, NX - 1):
            T[j] += RHS[j]

        if n == 1:
            res0 = res
        reste.append(res)

        if n % ifre == 0 or (res / res0) < eps:
            print(f"Iteration temporelle {n}: res = {res}")

    # Calcul des erreurs
    errL2h = 0
    errH1h = 0
    semih2 = 0
    for j in range(1, NX - 1):
        Texx = (Tex[j + 1] - Tex[j - 1]) / (2 * dx)
        Tx = (T[j + 1] - T[j - 1]) / (2 * dx)
        errL2h += dx * (T[j] - Tex[j])**2
        errH1h += dx * (Tx - Texx)**2
        Txx = (Tex[j + 1] - 2 * Tex[j] + Tex[j - 1]) / (dx**2)
        semih2 += dx * Txx**2

    errorL2[iter] = errL2h
    erreurH1[iter] = errL2h + errH1h
    semiH2[iter] = semih2

    print(f"Erreur L2: {errL2h}, Erreur H1: {errH1h}")

# Tracé des erreurs
plt.figure(3)
plt.loglog(itertab, np.sqrt(errorL2), label="Erreur L2")
plt.loglog(itertab, np.sqrt(erreurH1), label="Erreur H1")
plt.xlabel("Taille du maillage $h$")
plt.ylabel("Erreur")
plt.title("Erreurs $L^2$ et $H^1$ en fonction de $h$")
plt.legend()
plt.grid(True)
plt.show()
