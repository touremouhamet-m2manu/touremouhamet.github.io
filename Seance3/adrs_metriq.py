import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve

# Paramètres
a = 0.5
b = 10
c = 3
L = 0
R = 1

# Définition de la fonction f(x)
def fct(x):
    return a * x**2 + b * x + c * np.sin(4 * np.pi * x) + 10 * np.exp(-100 * (x - 0.5)**2)

# Dérivée seconde de f(x)
def fct_xx(x):
    return 2 * a - c * (4 * np.pi)**2 * np.sin(4 * np.pi * x) + 10 * 40000 * (x - 0.5)**2 * np.exp(-100 * (x - 0.5)**2)

# Intégrale exacte (pour comparaison)
I = quad(fct, L, R)
print(f"Intégrale exacte : {I[0]:.6f}")

# Tracer la fonction
npt = 1000
X = np.linspace(L, R, npt)
Y = fct(X)
plt.figure(figsize=(12, 6))
plt.plot(X, Y, ".")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Fonction f(x)")
plt.show()

# Intégrer en Riemann avec le nombre de points d'intégration croissant
def integrate_riemann(fct, L, R, err_tol):
    n = 5300
    h = np.zeros(n)
    IR = np.zeros(n)
    for npt in range(5200, n):
        h[npt] = (R - L) / (npt - 1)
        IR[npt] = 0
        for j in range(npt):
            x = (j - 1) * h[npt]
            IR[npt] += h[npt] * fct(x)
        error = abs(I[0] - IR[npt])
        if error < err_tol:
            break
    return npt, error

# Intégrer en Lebesgue avec adaptation de maillage
def integrate_lebesgue_adaptive(fct, fct_xx, L, R, err_tol, max_points=10000):
    x = L
    integral = 0.0
    epsilon = 0.01
    hmin = (R - L) / 1000
    hmax = (R - L) / 3
    npt = 0
    errors = []
    NXs = []

    while x < R and npt < max_points:
        u = fct(x)
        uxx = fct_xx(x)
        metric = min(max(abs(uxx) / epsilon, 1 / hmax**2), 1 / hmin**2)
        hloc = min(np.sqrt(1. / metric), R - x)
        ue = u
        x += hloc
        u = fct(x)
        integral += hloc * (u + ue) / 2
        npt += 1

        # Estimation de l'erreur L2
        error = abs(I[0] - integral)
        errors.append(error)
        NXs.append(npt)

        # Critère mixte d'arrêt
        if error < err_tol and npt >= 100:  # Exemple de critère mixte
            break

    return integral, npt, error, errors, NXs

# Tracer NX(err) pour err=0.04, 0.02, 0.01, 0.005, 0.0025
err_tols = [0.04, 0.02, 0.01, 0.005, 0.0025]
NXs_lebesgue = []
errors_lebesgue = []

for err_tol in err_tols:
    _, npt, _, errs, NXs = integrate_lebesgue_adaptive(fct, fct_xx, L, R, err_tol)
    NXs_lebesgue.append(npt)
    errors_lebesgue.append(err_tol)

plt.figure(figsize=(12, 6))
plt.loglog(errors_lebesgue, NXs_lebesgue, 'bo-', label='Lebesgue')
plt.xlabel('Erreur')
plt.ylabel('Nombre de points de maillage (NX)')
plt.title('NX en fonction de l\'erreur')
plt.legend()
plt.grid(True)
plt.show()

# Exemple d'utilisation avec une tolérance d'erreur
err_tol = 0.01
integral_lebesgue, npt_lebesgue, error_lebesgue, _, _ = integrate_lebesgue_adaptive(fct, fct_xx, L, R, err_tol)
print(f"Intégrale de Lebesgue adaptative (npt={npt_lebesgue}) : {integral_lebesgue:.6f}, Erreur: {error_lebesgue:.6f}")
