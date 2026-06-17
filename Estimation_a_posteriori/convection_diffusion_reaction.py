import numpy as np
import matplotlib.pyplot as plt

# Paramètres
nx, ny = 50, 50  # Nombre de points en x et y
Lx, Ly = 1.0, 1.0  # Longueur du domaine en x et y
dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # Pas d'espace
x, y = np.linspace(0, Lx, nx), np.linspace(0, Ly, ny)  # Grille spatiale
X, Y = np.meshgrid(x, y)

# Paramètres physiques
v1, v2 = 1.0, 1.0  # Composantes du vecteur vitesse V
nu = 0.01  # Coefficient de diffusion
lambda_ = 0.1  # Coefficient de réaction
Tc, k = 1.0, 10.0  # Paramètres de la source
sc = np.array([0.5, 0.5])  # Centre de la source

# Paramètres temporels
T = 1.0  # Temps total
dt = 0.001  # Pas de temps
nt = int(T / dt)  # Nombre de pas de temps

# Fonction source
def f(t, x, y):
    d_squared = (x - sc[0])**2 + (y - sc[1])**2
    return Tc * np.exp(-k * d_squared)

# Initialisation de la solution
u = np.zeros((ny, nx))

# Schéma d'Euler explicite avec différences finies
for _ in range(nt):
    un = u.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            uxx = (un[j, i+1] - 2 * un[j, i] + un[j, i-1]) / dx**2
            uyy = (un[j+1, i] - 2 * un[j, i] + un[j-1, i]) / dy**2
            ux = (un[j, i+1] - un[j, i-1]) / (2 * dx)
            uy = (un[j+1, i] - un[j-1, i]) / (2 * dy)
            u[j, i] = (un[j, i] + dt * (-v1 * ux - v2 * uy + nu * (uxx + uyy) - lambda_ * un[j, i] + f(_, x[i], y[j])))

    # Conditions aux limites de Dirichlet sur les bords entrants
    # Exemple : bord gauche (x=0) si v1 < 0
    if v1 < 0:
        u[:, 0] = 0
    if v1 > 0:
        u[:, -1] = 0
    if v2 < 0:
        u[0, :] = 0
    if v2 > 0:
        u[-1, :] = 0

# Solution exacte (pour comparaison, ici on utilise une solution fictive)
u_exact = np.exp(-((X - sc[0])**2 + (Y - sc[1])**2))

# Calcul des erreurs L2 et de la norme du gradient
L2_error = np.sqrt(np.mean((u - u_exact)**2))
gradient_x, gradient_y = np.gradient(u - u_exact, dx, dy)
gradient_norm_error = np.sqrt(np.mean(gradient_x**2 + gradient_y**2))

# Tracé des résultats
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.contourf(X, Y, u, levels=20, cmap='viridis')
plt.colorbar()
plt.title('Solution numérique')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 3, 2)
plt.contourf(X, Y, u_exact, levels=20, cmap='viridis')
plt.colorbar()
plt.title('Solution exacte')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 3, 3)
plt.contourf(X, Y, u - u_exact, levels=20, cmap='coolwarm')
plt.colorbar()
plt.title(f'Erreur (L2: {L2_error:.4f}, Gradient: {gradient_norm_error:.4f})')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()

