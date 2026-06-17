# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 18:18:11 2025

@author: SCD UM
"""

"""
adrs_seance2.py
Séance 2 : étude advection-diffusion 1D, source construite à partir d'une u_ex gaussienne,
implémentation upwind vs viscosite numérique, CFL, Neumann en sortie, calcul d'erreurs L2/H1,
tracés et étude de convergence sur plusieurs maillages.

Compatible Spyder / Jupyter.
"""
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Paramètres physiques
# -------------------------
L = 1.0
v = 1.0           # advection speed
nu = 0.01         # diffusion coefficient
lam = 1.0         # reaction coefficient

# Critères numériques / de simulation
max_time_steps = 2000000
tol_stationary = 1e-8        # critère pour ||u^{n+1}-u^n||_L2 (non normalisé)
tol_stationary_rel = 1e-8    # ou critère relatif
max_iter_inner = 200000

# -------------------------
# Fonctions analytiques
# -------------------------
def u_exact(s):
    """Solution exacte stationnaire (spatiale) choisie"""
    return np.exp(-10.0 * (s - L/2.0)**2)

def u_exact_derivative(s):
    """du/ds exact"""
    return (-20.0 * (s - L/2.0)) * np.exp(-10.0 * (s - L/2.0)**2)

def u_exact_second(s):
    """d2u/ds2 exact"""
    # derivative of (-20*(s-L/2)) * exp(...)
    x = s - L/2.0
    return (400.0 * x**2 - 20.0) * np.exp(-10.0 * x**2)

def compute_f_from_uex(s, v=v, nu=nu, lam=lam):
    """Given u_ex(s) (time-independent), compute f(s) = v u_s - nu u_ss + lam u."""
    return v * u_exact_derivative(s) - nu * u_exact_second(s) + lam * u_exact(s)

# -------------------------
# Discrétisations et utilitaires
# -------------------------
def make_mesh(NX):
    x = np.linspace(0.0, L, NX)
    dx = x[1] - x[0]
    return x, dx

def L2_norm_vec(e, dx):
    return np.sqrt(np.sum(e**2) * dx)

def H1_seminorm_vec(u_num, u_ex_grad, dx):
    # approximate derivative (central) for u_num
    dudx_num = np.zeros_like(u_num)
    dudx_num[1:-1] = (u_num[2:] - u_num[:-2]) / (2*dx)
    # for boundaries use one-sided
    dudx_num[0] = (u_num[1] - u_num[0]) / dx
    dudx_num[-1] = (u_num[-1] - u_num[-2]) / dx
    e_grad = dudx_num - u_ex_grad
    return np.sqrt(np.sum(e_grad**2) * dx)

# -------------------------
# Solveur 1D (explicit forward Euler)
# Options for advection: 'central+viscnum' or 'upwind'
# -------------------------
def solve_adrs_1d(NX, method='upwind', CFL_factor=0.45, verbose=False, max_iter=200000):
    """
    Solve time-dependent equation until stationarity:
    u_t + v u_x - nu u_xx + lam u = f(x)
    using explicit time stepping.
    Returns u (stationary), x, diagnostics dict with residual history.
    method: 'upwind' or 'central_viscnum'
    CFL_factor: safety factor (<0.5 recommended)
    """
    x, dx = make_mesh(NX)
    # discrete f on mesh using analytic formula
    f = compute_f_from_uex(x, v=v, nu=nu, lam=lam)

    # initial condition: choose u0 = 0 or u_exact to test consistency
    u = np.zeros(NX)
    # impose Dirichlet left to match u_exact(0) for compatibility:
    u[0] = u_exact(0.0)

    # choose dt from CFL-like criterion (safe)
    # heuristic dt <= C * dx^2 / (|v| dx + 2*nu + |f_max| dx^2)
    fmax = np.max(np.abs(f)) if np.any(f != 0) else 0.0
    dt_cfl = 0.5 * dx**2 / (abs(v) * dx + 2*nu + fmax * dx**2 + 1e-16)
    dt = CFL_factor * dt_cfl
    if verbose:
        print(f"NX={NX}, dx={dx:.3e}, dt_cfl={dt_cfl:.3e}, dt={dt:.3e}, fmax={fmax:.3e}")

    residuals = []
    norm0 = None
    for n in range(max_iter):
        u_old = u.copy()
        # interior updates
        # compute spatial derivatives depending on method
        if method == 'central_viscnum':
            # central first derivative and add numerical viscosity = 0.5*dx*|v| term in diffusion
            # we'll implement xnu = nu + 0.5*dx*|v|
            xnu = nu + 0.5 * dx * abs(v)
            # central differences
            ux = np.zeros_like(u)
            uxx = np.zeros_like(u)
            ux[1:-1] = (u_old[2:] - u_old[:-2]) / (2*dx)
            uxx[1:-1] = (u_old[2:] - 2*u_old[1:-1] + u_old[:-2]) / dx**2
            # update interior points
            u[1:-1] = u_old[1:-1] + dt * (-v * ux[1:-1] + xnu * uxx[1:-1] - lam * u_old[1:-1] + f[1:-1])
        elif method == 'upwind':
            # upwind for advection: if v>0 use (u_j - u_{j-1})/dx, else (u_{j+1} - u_j)/dx
            ux_up = np.zeros_like(u)
            if v >= 0:
                ux_up[1:] = (u_old[1:] - u_old[:-1]) / dx
                ux_up[0] = (u_old[1] - u_old[0]) / dx
            else:
                ux_up[:-1] = (u_old[1:] - u_old[:-1]) / dx
                ux_up[-1] = (u_old[-1] - u_old[-2]) / dx
            # second derivative central
            uxx = np.zeros_like(u)
            uxx[1:-1] = (u_old[2:] - 2*u_old[1:-1] + u_old[:-2]) / dx**2
            u[1:-1] = u_old[1:-1] + dt * (-v * ux_up[1:-1] + nu * uxx[1:-1] - lam * u_old[1:-1] + f[1:-1])
        else:
            raise ValueError("method must be 'upwind' or 'central_viscnum'")

        # -------------------------
        # Boundary conditions
        # Left: Dirichlet (compatible)
        u[0] = u_exact(0.0)
        # Right: Neumann u_x(L) = 0  -> enforce u[-1] = u[-2] (zero slope)
        # more accurate: linear extrapolation u[-1] = u[-2] (first order)
        u[-1] = u[-2]

        # compute residual ||u^{n+1} - u^n||_L2
        res = L2_norm_vec(u - u_old, dx)
        if norm0 is None:
            norm0 = L2_norm_vec(u_old, dx) + 1e-16
        residuals.append(res / norm0)
        # exit if stationary
        if res / norm0 < tol_stationary_rel or res < tol_stationary:
            if verbose:
                print(f"Converged at step {n}, res={res:.3e}, res_rel={res/norm0:.3e}")
            break
        # safety break
        if n == max_iter - 1:
            print("Warning: max iterations reached without convergence")

    diagnostics = {'residuals': np.array(residuals), 'dt': dt, 'iters': n+1}
    return u, x, f, diagnostics

# -------------------------
# Function to compute discrete L2 and H1 errors versus exact
# -------------------------
def compute_errors(u_num, x):
    dx = x[1] - x[0]
    u_ex = u_exact(x)
    e = u_num - u_ex
    L2 = L2_norm_vec(e, dx)
    # gradient exact
    u_ex_grad = u_exact_derivative(x)
    H1_semi = H1_seminorm_vec(u_num, u_ex_grad, dx)
    H1 = np.sqrt(L2**2 + H1_semi**2)
    return L2, H1, e

# -------------------------
# Routine principale : étude pour un maillage fixe NX=100, puis étude convergence sur 5 maillages
# -------------------------
if __name__ == "__main__":
    # ---------- Part A : fixed mesh NX=100 ----------
    NX_fixed = 100
    u_num_up, x_up, f_up, diag_up = solve_adrs_1d(NX_fixed, method='upwind', verbose=True, max_iter=max_iter_inner)
    u_num_cv, x_cv, f_cv, diag_cv = solve_adrs_1d(NX_fixed, method='central_viscnum', verbose=True, max_iter=max_iter_inner)

    # compute errors
    L2_up, H1_up, e_up = compute_errors(u_num_up, x_up)
    L2_cv, H1_cv, e_cv = compute_errors(u_num_cv, x_cv)
    print(f"[NX={NX_fixed}] Upwind: L2={L2_up:.3e}, H1={H1_up:.3e}, iters={diag_up['iters']}, dt={diag_up['dt']:.3e}")
    print(f"[NX={NX_fixed}] Cent+visc: L2={L2_cv:.3e}, H1={H1_cv:.3e}, iters={diag_cv['iters']}, dt={diag_cv['dt']:.3e}")

    # Save figure: solution vs exact
    plt.figure(figsize=(8,4))
    plt.plot(x_up, u_exact(x_up), 'k-', lw=2, label='u_exact')
    plt.plot(x_up, u_num_up, 'r--', lw=1, label='u_num (upwind)')
    plt.plot(x_cv, u_num_cv, 'b:', lw=1, label='u_num (central+visc)')
    plt.xlabel('s'); plt.ylabel('u(s)')
    plt.title(f'Solution stationnaire (NX={NX_fixed})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'solution_vs_exact_NX{NX_fixed}.png', dpi=300)
    plt.close()

    # Residual evolution
    plt.figure(figsize=(8,4))
    plt.semilogy(diag_up['residuals'], label='upwind residual (rel)')
    plt.semilogy(diag_cv['residuals'], label='cent+visc residual (rel)')
    plt.xlabel('iteration')
    plt.ylabel('residual (||u^{n+1}-u^n|| / ||u^0||)')
    plt.title('Convergence temporelle vers la stationnarité')
    plt.legend(); plt.grid(True)
    plt.savefig(f'convergence_norm_NX{NX_fixed}.png', dpi=300)
    plt.close()

    # ---------- Part B : convergence h -> errors on 5 meshes ----------
    # Start from NX=3 and generate 5 meshes increasing
    NX_list = [3, 5, 9, 17, 33]   # 5 maillages (start=3)
    L2_list = []
    H1_list = []
    h_list = []
    for NX in NX_list:
        u_num, x, f_num, diag = solve_adrs_1d(NX, method='upwind', CFL_factor=0.45, verbose=False, max_iter=200000)
        L2, H1, _ = compute_errors(u_num, x)
        L2_list.append(L2)
        H1_list.append(H1)
        h_list.append(x[1] - x[0])
        print(f"NX={NX}, h={x[1]-x[0]:.3e}, L2={L2:.3e}, H1={H1:.3e}, iters={diag['iters']}")

    # Plot errors vs h (log-log)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.loglog(h_list, L2_list, 'o-', label='L2 error')
    # for reference plot h and h^2 slopes (choose scaling)
    Cref = L2_list[0] / (h_list[0]**1)
    plt.loglog(h_list, [Cref * (h**1) for h in h_list], '--', label='~h^1')
    plt.loglog(h_list, [Cref * (h**2) for h in h_list], '-.', label='~h^2')
    plt.xlabel('h (dx)'); plt.ylabel('L2 error'); plt.title('L2 error vs h')
    plt.legend(); plt.grid(True)
    plt.show()

    plt.subplot(1,2,2)
    plt.loglog(h_list, H1_list, 's-', label='H1 error')
    Cref2 = H1_list[0] / (h_list[0]**1)
    plt.loglog(h_list, [Cref2 * (h**1) for h in h_list], '--', label='~h^1')
    plt.xlabel('h (dx)'); plt.ylabel('H1 error'); plt.title('H1 error vs h')
    plt.legend(); plt.grid(True)
    plt.show()

    plt.tight_layout()
    plt.savefig('errors_vs_h.png', dpi=300)
    plt.close()

    print("Figures sauvegardées : solution_vs_exact_..., convergence_norm_..., errors_vs_h.png")
