import numpy as np
import matplotlib.pyplot as plt

# Paramètres
nx = 100
L = 1.0
dx = L / (nx - 1)
x = np.linspace(0, L, nx)
v1 = 1.0
nu = 0.01
lambda_ = 0.1
T = 1.0
dt = 0.005
nt = int(T / dt)

# Conditions aux limites
ul = 0.0  # Valeur de Dirichlet à gauche
g = 1.0   # Valeur de Neumann à droite

# Condition initiale compatible avec les conditions aux limites
u0 = ul + g * x

# Fonction source
def f(t, x):
    return np.sin(np.pi * x / L)  # Exemple de fonction source

# Initialisation de la solution
u = u0.copy()

# Schéma d'Euler explicite avec différences finies
for n in range(nt):
    un = u.copy()
    for i in range(1, nx - 1):
        uxx = (un[i+1] - 2 * un[i] + un[i-1]) / dx**2
        ux = (un[i+1] - un[i-1]) / (2 * dx)
        u[i] = un[i] + dt * (-v1 * ux + nu * uxx - lambda_ * un[i] + f(n * dt, x[i]))

    # Conditions aux limites
    u[0] = ul  # Dirichlet à gauche
    u[-1] = u[-2] + g * dx  # Approximation de Neumann à droite

# Tracé des résultats
plt.figure(figsize=(12, 5))
plt.plot(x, u, label='Solution numérique')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution numérique de l\'équation 1D')
plt.legend()
plt.grid(True)
plt.show()

