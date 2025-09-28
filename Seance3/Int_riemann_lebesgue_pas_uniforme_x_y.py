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
def f(x):
    return a * x**2 + b * x + c * np.sin(4 * np.pi * x) + 10 * np.exp(-100 * (x - 0.5)**2)

# Dérivée seconde de f(x)
def f_xx(x):
    return 2*a - c*(4*np.pi)**2*np.sin(4*np.pi*x) + 10*40000*(x-0.5)**2*np.exp(-100*(x-0.5)**2)

# Intégrale exacte (pour comparaison)
I_exact, _ = quad(f, L, R)
print(f"Intégrale exacte : {I_exact:.6f}")

# Tracer la fonction
x_plot = np.linspace(L, R, 1000)
y_plot = f(x_plot)
plt.figure(figsize=(12, 6))
plt.plot(x_plot, y_plot, label="$f(x)$")
plt.title("Fonction $f(x)$")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.grid(True)
plt.legend()
plt.show()

# 1. Intégration de Riemann (pas uniforme en x)
def integrate_riemann(f, L, R, precision=1.e-3):
    n_max = 100000
    for n in range(100, n_max):
        x = np.linspace(L, R, n, endpoint=False)
        dx = (R - L) / n
        integral = np.sum(f(x) * dx)
        error = abs(integral - I_exact)
        if error < precision:
            print(f"Intégrale de Riemann (n={n}) : {integral:.6f}, Erreur: {error:.6f}")
            return integral, n, error
    print("Précision non atteinte pour Riemann.")
    return None, None, None

riemann_integral, n_riemann, error_riemann = integrate_riemann(f, L, R)

# 2.a. Intégration de Lebesgue (pas uniforme en y)
def integrate_lebesgue(f, L, R, N):
    x = np.linspace(L, R, 10000)
    y = f(x)
    y_min = np.min(y)
    y_max = np.max(y)
    dy = (y_max - y_min) / (N - 1)
    y_values = np.linspace(y_min, y_max, N)

    def find_x_for_y(y_target):
        def equation(x):
            return f(x) - y_target
        x_guesses = np.linspace(L, R, 10)
        solutions = []
        for x_guess in x_guesses:
            try:
                sol, = fsolve(equation, x_guess)
                if L <= sol <= R:
                    solutions.append(sol)
            except RuntimeError:
                pass
        solutions = np.unique(np.round(solutions, decimals=6))
        return solutions

    integral = 0.0
    for j in range(N - 1):
        y_j = y_values[j]
        y_j1 = y_values[j + 1]
        x_j = find_x_for_y(y_j)
        x_j1 = find_x_for_y(y_j1)

        if len(x_j) > 0 and len(x_j1) > 0:
            x_j_mean = np.mean(x_j)
            x_j1_mean = np.mean(x_j1)
            integral += (x_j1_mean - x_j_mean) * dy

    return integral

def find_lebesgue_N(f, L, R, precision=1.e-3):
    N = 100
    while True:
        integral = integrate_lebesgue(f, L, R, N)
        error = abs(integral - I_exact)
        if error < precision:
            print(f"Intégrale de Lebesgue (N={N}) : {integral:.6f}, Erreur: {error:.6f}")
            return integral, N, error
        N += 100

lebesgue_integral, N_lebesgue, error_lebesgue = find_lebesgue_N(f, L, R)

# 2.b. Suite de Cauchy pour Lebesgue
def integrate_lebesgue_cauchy(f, L, R, N, n_points_y):
    x = np.linspace(L, R, n_points_y)
    y = f(x)
    y_min = np.min(y)
    y_max = np.max(y)
    dy = (y_max - y_min) / (N - 1)
    y_values = np.linspace(y_min, y_max, N)

    def find_x_for_y(y_target):
        def equation(x):
            return f(x) - y_target
        x_guesses = np.linspace(L, R, 10)
        solutions = []
        for x_guess in x_guesses:
            try:
                sol, = fsolve(equation, x_guess)
                if L <= sol <= R:
                    solutions.append(sol)
            except RuntimeError:
                pass
        solutions = np.unique(np.round(solutions, decimals=6))
        return solutions

    integral = 0.0
    for j in range(N - 1):
        y_j = y_values[j]
        y_j1 = y_values[j + 1]
        x_j = find_x_for_y(y_j)
        x_j1 = find_x_for_y(y_j1)

        if len(x_j) > 0 and len(x_j1) > 0:
            x_j_mean = np.mean(x_j)
            x_j1_mean = np.mean(x_j1)
            integral += (x_j1_mean - x_j_mean) * dy

    return integral

def find_lebesgue_N_cauchy(f, L, R, precision=1.e-3):
    N = 100
    n_points_y = 1000
    resultat_old = 0
    while True:
        resultat = integrate_lebesgue_cauchy(f, L, R, N, n_points_y)
        error = abs(resultat - I_exact)
        var_cauchy = abs(resultat - resultat_old)
        if error < precision and var_cauchy < precision:
            print(f"Intégrale de Lebesgue avec Cauchy (N={N}, n_points_y={n_points_y}) : {resultat:.6f}, Erreur: {error:.6f}, Var Cauchy: {var_cauchy:.6f}")
            return resultat, N, n_points_y, error, var_cauchy
        resultat_old = resultat
        N += 100
        n_points_y += 100

lebesgue_cauchy_integral, N_lebesgue_cauchy, n_points_y_lebesgue_cauchy, error_lebesgue_cauchy, var_cauchy = find_lebesgue_N_cauchy(f, L, R)

# 3. Adaptation du pas d'intégration (contrôle de métrique)
def integrate_adaptive(f, f_xx, L, R, precision=1.e-3):
    x = L
    integral = 0.0
    epsilon = 0.01
    hmin = (R - L) / 1000
    hmax = (R - L) / 10
    npt = 0

    while x < R:
        u = f(x)
        uxx = f_xx(x)
        metric = min(max(abs(uxx) / epsilon, 1 / hmax**2), 1 / hmin**2)
        hloc = min(np.sqrt(1. / metric), R - x)
        ue = u
        x += hloc
        u = f(x)
        integral += hloc * (u + ue) / 2
        npt += 1

    error = abs(integral - I_exact)
    print(f"Intégrale adaptative (npt={npt}) : {integral:.6f}, Erreur: {error:.6f}")
    return integral, npt, error

adaptive_integral, npt_adaptive, error_adaptive = integrate_adaptive(f, f_xx, L, R)

# Tracer les erreurs
plt.figure(figsize=(12, 6))
plt.loglog(n_riemann, error_riemann, 'b-', label='Riemann')
plt.loglog(N_lebesgue, error_lebesgue, 'r-', label='Lebesgue')
plt.loglog(N_lebesgue_cauchy, error_lebesgue_cauchy, 'g-', label='Lebesgue avec Cauchy')
plt.loglog(npt_adaptive, error_adaptive, 'm-', label='Adaptative')
plt.xlabel('Nombre d\'évaluations de la fonction')
plt.ylabel('Erreur absolue')
plt.title('Erreurs d\'intégration en fonction du nombre d\'évaluations')
plt.legend()
plt.grid(True)
plt.show()
