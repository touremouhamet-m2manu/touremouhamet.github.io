import numpy as np
import matplotlib.pyplot as plt

# Paramètres
lambda_ = 1.0
u0 = 1.0
T = 60  # 1 minute en secondes
dt_values = np.logspace(0, -3, 20)  # Pas de temps décroissants de 1s à 0.001s

# Solution exacte
def u_exact(t):
    return u0 * np.exp(-lambda_ * t)

# Schéma d'Euler explicite avec différences finies
def euler_explicit(lambda_, u0, dt, T):
    nt = int(T / dt)
    u = np.zeros(nt + 1)
    u[0] = u0
    for n in range(nt):
        u[n+1] = u[n] + dt * (-lambda_ * u[n])
    return u

# Tracer la solution exacte et numérique pour Δt = 1s
dt_1s = 1.0
u_history_1s = euler_explicit(lambda_, u0, dt_1s, T)
t_1s = np.arange(0, T + dt_1s, dt_1s)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_1s, [u_exact(t) for t in t_1s], label='Solution exacte')
plt.plot(t_1s, u_history_1s, label='Solution numérique (Δt=1s)')
plt.xlabel('Temps (s)')
plt.ylabel('u(t)')
plt.title('Comparaison solution exacte et numérique')
plt.legend()
plt.grid(True)

# Calcul des erreurs L2 et de la dérivée pour les 20 pas de temps
L2_errors = []
derivative_errors = []

for dt in dt_values:
    u_history = euler_explicit(lambda_, u0, dt, T)
    t = np.linspace(0, T, len(u_history))
    u_exact_values = u_exact(t)
    L2_error = np.sqrt(np.mean((u_history - u_exact_values)**2))
    L2_errors.append(L2_error)

    # Calcul de la dérivée de l'erreur
    error = u_history - u_exact_values
    derivative_error = np.sqrt(np.mean(np.gradient(error, dt)**2))
    derivative_errors.append(derivative_error)

# Tracer les erreurs L2 et de la dérivée en fonction du pas de temps
plt.subplot(1, 2, 2)
plt.loglog(dt_values, L2_errors, marker='o', label='Erreur L2')
plt.loglog(dt_values, derivative_errors, marker='o', color='r', label="Erreur sur la dérivée")
plt.xlabel('Pas de temps (Δt)')
plt.ylabel('Erreur')
plt.title('Erreurs L2 et sur la dérivée en fonction du pas de temps')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

