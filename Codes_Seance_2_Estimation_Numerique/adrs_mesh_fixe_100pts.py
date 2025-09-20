import numpy as np
import matplotlib.pyplot as plt

# PARAMÈTRES PHYSIQUES
K = 0.01  # Coefficient de diffusion (nu)
L = 1.0   # Taille du domaine
Temps = 20.0  # Temps d'intégration
V = 1.0
lambda_ = 1.0

# PARAMÈTRES NUMÉRIQUES
NX = 100  # Nombre de points de grille
NT = 10000  # Nombre de pas de temps max
ifre = 10  # Tracer toutes les itérations de temps ifre
eps = 0.001  # Rapport de convergence relative

# Solution exacte et fonction source f(s)
def u_ex(s):
    return np.exp(-10 * (s - L/2)**2)

def f(s):
    u = u_ex(s)
    us = -20 * (s - L/2) * u
    uss = (-20 + 400 * (s - L/2)**2) * u
    return V * us - K * uss + lambda_ * u

# Initialisation
x = np.linspace(0.0, L, NX)
dx = L / (NX - 1)
dt = dx**2 / (V * dx + 2 * K + np.max(np.abs(f(x))) * dx**2)

T = u_ex(x)
F = f(x)

# Imposer u_s(L) = 0 dès l'initialisation
T[-1] = T[-2]

# Boucle principale en temps
n = 0
res = 1
res0 = 1
reste = []

while n < NT and res / res0 > eps:
    n += 1
    res = 0
    RHS = np.zeros(NX)

    for j in range(1, NX - 1):
        # Décentrage amont pour Tx
        if V > 0:
            Tx = (T[j] - T[j-1]) / dx
        else:
            Tx = (T[j+1] - T[j]) / dx
        Txx = (T[j-1] - 2 * T[j] + T[j+1]) / (dx**2)
        RHS[j] = dt * (-V * Tx + K * Txx - lambda_ * T[j] + F[j])
        res += abs(RHS[j])

    for j in range(1, NX - 1):
        T[j] += RHS[j]

    # Imposer u_s(L) = 0 après la mise à jour
    T[-1] = T[-2]

    if n == 1:
        res0 = res
    reste.append(res)

    if n % ifre == 0 or (res / res0) < eps:
        print(n, res)
        plt.plot(x, T, label=f"t = {n * dt:.2f}")

plt.xlabel('$x$')
plt.ylabel('$T$')
plt.title('Solution numérique avec $u_s(L)=0$')
plt.legend()
plt.show()
