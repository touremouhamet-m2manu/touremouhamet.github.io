import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

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
X = np.zeros(npt)
hh = (R - L) / (npt - 1)
for i in range(npt):
    x = (i - 1) * hh
    X[i-1] = x
Y = fct(X)
plt.plot(X, Y, ".")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Fonction f(x)")
plt.show()

# Intégrer en Riemann avec le nombre de points d'intégration croissant
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
    if error < 1.e-3:
        break
print(f"Intégrale de Riemann npt, error: {npt}, {error:.6f}")

# Intégrer en Lebesgue
hmin = (R - L) / 100
hmax = (R - L) / 3
epsilon = 0.01
itermax_lebesgue = 10
nptL = np.zeros(itermax_lebesgue, dtype=int)
eps = np.zeros(itermax_lebesgue)
IL = np.zeros(itermax_lebesgue)

for npt in range(itermax_lebesgue):
    x = L
    u = fct(x)
    uxx = fct_xx(x)
    metric = min(max(abs(uxx) / epsilon, 1 / hmax**2), 1 / hmin**2)
    hloc = min(np.sqrt(1. / metric), R - x)
    ue = u
    x += hloc
    u = fct(x)
    IL[npt] += hloc * (u + ue) / 2
    nptL[npt] += 1
    eps[npt] = epsilon
    error = abs(I[0] - IL[npt])
    if error < 1.e-3:
        break
print(f"Intégrale de Lebesgue epsilon, npt, error: {epsilon}, {nptL[npt]}, {error:.6f}")

# Tracer les erreurs
plt.plot(np.log10(abs(IR[5200:npt] - I[0])), label="Riemann")
plt.plot(np.log10(abs(IL - I[0])), '*--', label="Lebesgue")
plt.xlabel("ITER REFINEMENT")
plt.ylabel("Log10(Erreur)")
plt.legend()
plt.show()

# Convergence de Cauchy pour Lebesgue
resultat = 1
N = 10000
n_points_y = 3
var_cauchyt = [1]
err_result = [1]

for itest_lebesgue in range(30):
    resultat_old = resultat
    # Fonction pour l'intégrale de Lebesgue
    def integrale_lebesgue(fct, L, R, N, n_points_y):
        x = np.linspace(L, R, n_points_y)
        y = fct(x)
        y_min = np.min(y)
        y_max = np.max(y)
        levels = np.linspace(y_min, y_max, N)
        integral = 0.0
        for i in range(N - 1):
            y_i = levels[i]
            y_i1 = levels[i + 1]
            mask = (y >= y_i) & (y < y_i1)
            if np.any(mask):
                x_masked = x[mask]
                if len(x_masked) > 0:
                    measure = x_masked[-1] - x_masked[0]
                    integral += (y_i + y_i1) / 2 * measure
        return integral

    resultat = integrale_lebesgue(fct, L, R, N, n_points_y)
    err = abs(resultat - I[0])
    var_cauchy = abs(resultat - resultat_old)
    var_cauchyt.append(var_cauchy)
    err_result.append(err)
    print(f"Variation de Cauchy : {var_cauchy:.6f}")
    n_points_y += 100

err_result = np.array(err_result[1:])
var_cauchyt = np.array(var_cauchyt[1:])

plt.figure()
plt.semilogy(np.log10(err_result / err_result[0]), label='Erreur relative')
plt.semilogy(np.log10(var_cauchyt / var_cauchyt[0]), label='Variation de Cauchy')
plt.xlabel('Itération Cauchy')
plt.ylabel('Log10(Erreur relative/Variation de Cauchy)')
plt.legend()
plt.grid(True)
plt.show()
