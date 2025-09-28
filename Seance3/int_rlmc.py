import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve

# Définition de la fonction f(x)
def f(x):
    return 1 / np.sqrt(1 - x**2)

# Intégrale exacte
I_exact = np.pi
print(f"Intégrale exacte : {I_exact:.6f}")

# Méthode de Riemann
def integrate_riemann(f, L, R, N):
    x = np.linspace(L, R, N, endpoint=False)
    dx = (R - L) / N
    integral = np.sum(f(x) * dx)
    return integral

# Méthode de Lebesgue
def integrate_lebesgue(f, L, R, N):
    # Éviter les singularités aux bornes
    x = np.linspace(L + 1e-10, R - 1e-10, 10000)
    y = f(x)
    y_min = np.min(y)
    y_max = np.max(y)
    dy = (y_max - y_min) / (N - 1)
    y_values = np.linspace(y_min, y_max, N)

    def find_x_for_y(y_target):
        def equation(x):
            return f(x) - y_target
        x_guesses = np.linspace(L + 1e-10, R - 1e-10, 10)
        solutions = []
        for x_guess in x_guesses:
            try:
                sol, = fsolve(equation, x_guess)
                if L < sol < R:
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

# Méthode de Monte Carlo
def integrate_monte_carlo(f, L, R, N):
    x_random = np.random.uniform(L + 1e-10, R - 1e-10, N)
    integral = (R - L) * np.mean(f(x_random))
    return integral

# Fonction pour calculer les erreurs en fonction de N
def compute_errors(max_N, method, method_name):
    N_values = np.logspace(2, 5, 10, dtype=int)
    errors = []
    for N in N_values:
        integral = method(f, -1, 1, N)
        error = abs(integral - I_exact)
        errors.append(error)
    return N_values, errors

# Calcul des erreurs pour chaque méthode
N_riemann, errors_riemann = compute_errors(100000, integrate_riemann, "Riemann")
N_lebesgue, errors_lebesgue = compute_errors(10000, integrate_lebesgue, "Lebesgue")
N_monte_carlo = np.logspace(3, 6, 10, dtype=int)
errors_monte_carlo = []
for N in N_monte_carlo:
    integral = integrate_monte_carlo(f, -1, 1, N)
    error = abs(integral - I_exact)
    errors_monte_carlo.append(error)

# Tracé des courbes d'erreurs
plt.figure(figsize=(12, 6))
plt.loglog(N_riemann, errors_riemann, 'b-', label='Riemann')
plt.loglog(N_lebesgue, errors_lebesgue, 'r-', label='Lebesgue')
plt.loglog(N_monte_carlo, errors_monte_carlo, 'g-', label='Monte Carlo')
plt.xlabel('Nombre d\'évaluations de la fonction')
plt.ylabel('Erreur absolue')
plt.title('Erreurs d\'intégration en fonction du nombre d\'évaluations')
plt.legend()
plt.grid(True)
plt.show()

# Trouver le N minimal pour une précision de 1.e-3
def find_minimal_N(integrate_func, precision=1.e-3):
    N = 100
    while True:
        integral = integrate_func(f, -1, 1, N)
        error = abs(integral - I_exact)
        if error < precision:
            print(f"{integrate_func.__name__} (N={N}) : {integral:.6f}, Erreur: {error:.6f}")
            return N, error
        N += 100

N_riemann_min, error_riemann_min = find_minimal_N(integrate_riemann)
N_lebesgue_min, error_lebesgue_min = find_minimal_N(integrate_lebesgue)

# Méthode de Monte Carlo avec précision
def find_minimal_N_monte_carlo(precision=1.e-3):
    N = 1000
    while True:
        integral = integrate_monte_carlo(f, -1, 1, N)
        error = abs(integral - I_exact)
        if error < precision:
            print(f"Monte Carlo (N={N}) : {integral:.6f}, Erreur: {error:.6f}")
            return N, error
        N += 1000

N_monte_carlo_min, error_monte_carlo_min = find_minimal_N_monte_carlo()
