# Euler_ODE_Errors_fixed.py
import numpy as np
import matplotlib.pyplot as plt

def euler_explicit(lambda_, u0, dt, T):
    N = int(np.floor(T / dt)) + 1
    t = np.linspace(0, N*dt, N)
    u = np.zeros_like(t)
    u[0] = u0
    for n in range(N-1):
        u[n+1] = u[n] + dt * (-lambda_ * u[n])
    return t, u

def exact_solution(lambda_, u0, t):
    return u0 * np.exp(-lambda_ * t)

def compute_L2_error(u_num, u_ex, dt):
    e = u_num - u_ex
    return np.sqrt(np.sum(e**2) * dt)

# Parameters
lambda_ = 1.0
u0 = 1.0
T = 60.0   # 1 minute
dt_ref = 1.0  # trace for Dt=1s

# 1) Time trace for Dt = 1s
t1, u_num1 = euler_explicit(lambda_, u0, dt_ref, T)
u_ex1 = exact_solution(lambda_, u0, t1)

plt.figure(figsize=(8,4))
plt.plot(t1, u_ex1, 'k-', linewidth=2, label='Solution exacte')
plt.plot(t1, u_num1, 'ro', markersize=4, label=f'Euler explicite dt={dt_ref}s')
plt.xlabel('Temps (s)')
plt.ylabel('u(t)')
plt.title('Euler explicite vs solution exacte (dt=1s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('euler_solution.png', dpi=300)
plt.close()

# 2) Convergence en dt : 20 pas de temps décroissants de 1s à 0.001s
# j'utilise geomspace puis je convertis en array
dts = np.geomspace(1.0, 0.001, 20)
errors_u = []
errors_du = []

for dt in dts:
    t, u_num = euler_explicit(lambda_, u0, dt, T)
    u_ex = exact_solution(lambda_, u0, t)
    # L2 error on u
    err_u = compute_L2_error(u_num, u_ex, dt)
    errors_u.append(err_u)
    # approximate time derivative: forward difference / dt -> compare to exact -lambda u at midpoints
    if len(u_num) > 1:
        u_num_dt = np.diff(u_num) / dt
        t_mid = t[:-1] + dt/2
        u_ex_dt_mid = -lambda_ * exact_solution(lambda_, u0, t_mid)
        err_du = np.sqrt(np.sum((u_num_dt - u_ex_dt_mid)**2) * dt)
    else:
        err_du = 0.0
    errors_du.append(err_du)

# Plot errors vs dt (log-log)
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.loglog(dts, errors_u, 'o-')
plt.xlabel('dt (s)')
plt.ylabel('Erreur L2 sur u')
plt.title('Erreur L2(u) en fonction du pas temporel')
plt.grid(True, which='both', ls='--')

plt.subplot(1,2,2)
plt.loglog(dts, errors_du, 's-')
plt.xlabel('dt (s)')
plt.ylabel('Erreur L2 sur du/dt')
plt.title('Erreur L2(du/dt) en fonction du pas temporel')
plt.grid(True, which='both', ls='--')

plt.tight_layout()
plt.savefig('euler_errors.png', dpi=300)
plt.close()

print("Figures sauvegardées : euler_solution.png, euler_errors.png")
