import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Équation: u,t + V u,x - K u,xx = -λ u + f
# Conditions aux limites: u(0,t) = ul, u,x(L,t) = g

# PARAMÈTRES PHYSIQUES
L = 1.0           # Longueur du domaine
Time = 1.0        # Temps final
V = 0.5           # Vitesse d'advection
K = 0.01          # Coefficient de diffusion
lamda = 0.1       # Coefficient de réaction

# CONDITIONS AUX LIMITES
ul = 1.0          # Dirichlet à gauche: u(0,t) = ul
g = 0.0           # Neumann à droite: u,x(L,t) = g

# PARAMÈTRES NUMÉRIQUES
NX = 50           # Nombre de points de grille
NT = 1000         # Nombre de pas de temps
eps = 1e-6        # Critère de convergence

def source_term(x, t):
    """Terme source gaussien"""
    Tc = 1.0
    xc = 0.3      # Centre de la source
    k = 100.0     # Largeur de la source
    return Tc * np.exp(-k * (x - xc)**2) * np.exp(-0.5*t)

def initial_condition(x):
    """Condition initiale compatible avec les CL"""
    # u0(x) = ul + g*x (solution linéaire compatible) + petite perturbation
    return ul + g * x + 0.1 * np.sin(2*np.pi*x/L)

def exact_solution(x, t):
    """Solution exacte pour validation (si disponible)"""
    # Pour un cas test simple
    return np.exp(-lamda*t) * (ul + g*x + 0.1*np.sin(2*np.pi*x/L))

def solve_pde_1d():
    """Résolution de l'EDP 1D avec conditions mixtes"""
    
    dx = L / (NX - 1)
    dt = min(dx**2/(2*K), dx/abs(V), 0.01)  # Condition CFL
    print(f"dx = {dx:.4f}, dt = {dt:.6f}")
    
    # Grille spatiale
    x = np.linspace(0, L, NX)
    
    # Initialisation
    u = initial_condition(x)
    u_new = u.copy()
    
    # Historique pour visualisation
    u_history = [u.copy()]
    time_points = [0.0]
    
    # Boucle temporelle
    t = 0.0
    for n in range(NT):
        if t >= Time:
            break
            
        # Construction de la matrice (format LIL pour modification facile)
        A = lil_matrix((NX, NX))
        b = np.zeros(NX)
        
        for i in range(NX):
            if i == 0:
                # Condition de Dirichlet à gauche: u(0,t) = ul
                A[i, i] = 1.0
                b[i] = ul
                
            elif i == NX - 1:
                # Condition de Neumann à droite: u,x(L,t) = g
                # Schéma décentré d'ordre 1: (u_i - u_{i-1})/dx = g
                A[i, i-1] = -1.0/dx
                A[i, i] = 1.0/dx
                b[i] = g
                
            else:
                # Équation discrétisée (schéma implicite)
                # u,t + V u,x - K u,xx + λ u = f
                
                # Terme temporel: (u_new[i] - u_old[i])/dt
                A[i, i] = 1.0/dt + lamda
                
                # Terme d'advection: V u,x
                # Schéma décentré amont
                if V > 0:
                    A[i, i-1] += V/dx
                    A[i, i] += -V/dx
                else:
                    A[i, i] += V/dx
                    A[i, i+1] += -V/dx
                
                # Terme de diffusion: -K u,xx
                A[i, i-1] += -K/dx**2
                A[i, i] += 2*K/dx**2
                A[i, i+1] += -K/dx**2
                
                # Terme source
                f_val = source_term(x[i], t)
                b[i] = u[i]/dt + f_val
        
        # Conversion en format CSR pour résolution efficace
        A_csr = A.tocsr()
        
        # Résolution du système linéaire
        u_new = spsolve(A_csr, b)
        u = u_new.copy()
        
        t += dt
        if n % 100 == 0 or n == NT-1:
            u_history.append(u.copy())
            time_points.append(t)
            print(f"Step {n}, t = {t:.3f}, max(u) = {np.max(u):.4f}")
    
    return x, np.array(u_history), time_points

def calculate_errors(u_final, u_exact, dx):
    """Calcul des erreurs L2 et H1"""
    # Erreur L2 sur la solution
    error_l2 = np.sqrt(np.sum((u_final - u_exact)**2) * dx)
    
    # Erreur sur le gradient (norme H1)
    grad_u = np.gradient(u_final, dx)
    grad_exact = np.gradient(u_exact, dx)
    error_h1 = np.sqrt(np.sum((grad_u - grad_exact)**2) * dx)
    
    return error_l2, error_h1

# Résolution du problème
print("Début de la résolution...")
x, u_history, time_points = solve_pde_1d()
u_final = u_history[-1]

# Solution exacte pour comparaison
u_exact = exact_solution(x, time_points[-1])

# Calcul des erreurs
dx = L / (NX - 1)
error_l2, error_h1 = calculate_errors(u_final, u_exact, dx)
print(f"Erreur L2: {error_l2:.6e}")
print(f"Erreur H1 (gradient): {error_h1:.6e}")

# Visualisation
plt.figure(figsize=(15, 5))

# Solution finale
plt.subplot(1, 3, 1)
plt.plot(x, u_final, 'b-', linewidth=2, label='Solution numérique')
plt.plot(x, u_exact, 'r--', linewidth=2, label='Solution exacte')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title(f'Solution à t = {time_points[-1]:.2f}')
plt.legend()
plt.grid(True)

# Erreur
plt.subplot(1, 3, 2)
error = np.abs(u_final - u_exact)
plt.plot(x, error, 'g-', linewidth=2)
plt.xlabel('x')
plt.ylabel('Erreur absolue')
plt.title(f'Erreur L2: {error_l2:.2e}')
plt.grid(True)

# Gradient
plt.subplot(1, 3, 3)
grad_u = np.gradient(u_final, dx)
grad_exact = np.gradient(u_exact, dx)
plt.plot(x, grad_u, 'b-', linewidth=2, label='Gradient numérique')
plt.plot(x, grad_exact, 'r--', linewidth=2, label='Gradient exact')
plt.xlabel('x')
plt.ylabel('du/dx')
plt.title('Gradient de la solution')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('pde_1d_mixed_bc_solution.png', dpi=300)
plt.show()

# Évolution temporelle
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(u_history)))
for i, (u_val, color) in enumerate(zip(u_history, colors)):
    if i % max(1, len(u_history)//10) == 0:  # Afficher environ 10 courbes
        plt.plot(x, u_val, color=color, alpha=0.7, label=f't = {time_points[i]:.2f}')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Évolution temporelle de la solution')
plt.legend()
plt.grid(True)
plt.savefig('pde_1d_time_evolution.png', dpi=300)
plt.show()

print("Calcul terminé avec succès!")
