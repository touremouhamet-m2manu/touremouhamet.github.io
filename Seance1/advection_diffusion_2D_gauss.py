# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 16:45:00 2025

@author: SCD UM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def solve_adrs_2d(Nx, Ny, dt, T, V=(0.5,0.0), nu=0.01, lam=0.1, Tc=1.0, k=200.0):
    Lx, Ly = 1.0, 1.0
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    dx = x[1]-x[0]; dy = y[1]-y[0]
    u = np.zeros((Nx,Ny))
    # source center
    sc = (Lx*0.5, Ly*0.5)
    X, Y = np.meshgrid(x,y,indexing='ij')
    # time stepping
    Nt = int(T/dt)
    for n in range(Nt):
        # compute source at time t (stationary amplitude)
        f = Tc * np.exp(-k * ((X-sc[0])**2 + (Y-sc[1])**2))
        u_new = u.copy()
        # interior points update (explicit)
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                ux = (u[i+1,j]-u[i-1,j])/(2*dx)
                uy = (u[i,j+1]-u[i,j-1])/(2*dy)
                uxx = (u[i+1,j]-2*u[i,j]+u[i-1,j])/(dx*dx)
                uyy = (u[i,j+1]-2*u[i,j]+u[i,j-1])/(dy*dy)
                adv = V[0]*ux + V[1]*uy
                diff = nu*(uxx+uyy)
                u_new[i,j] = u[i,j] + dt*(-adv + diff - lam*u[i,j] + f[i,j])
        # Boundary conditions simple: Dirichlet 0
        u_new[0,:] = 0.0
        u_new[-1,:] = 0.0
        u_new[:,0] = 0.0
        u_new[:,-1] = 0.0
        u = u_new
    return x,y,u

def compute_L2_error(u, uref, x, y):
    # assume same grid
    dx = x[1]-x[0]; dy = y[1]-y[0]
    diff = u - uref
    return np.sqrt(np.sum(diff**2) * dx * dy)

# Parameters (cheap so script runs fast)
Nx_coarse, Ny_coarse = 50, 50
Nx_fine, Ny_fine = 100, 100
T = 0.5
dt_coarse = 0.0005
dt_fine = 0.00025

# reference solution on finer grid (short time)
x_ref, y_ref, u_ref = solve_adrs_2d(Nx_fine, Ny_fine, dt_fine, T, V=(0.5,0.0), nu=0.01, lam=0.1)
# solve on coarse grid
x, y, u = solve_adrs_2d(Nx_coarse, Ny_coarse, dt_coarse, T, V=(0.5,0.0), nu=0.01, lam=0.1)

# interpolate reference to coarse grid
interp = RegularGridInterpolator((x_ref, y_ref), u_ref)
Xc, Yc = np.meshgrid(x,y,indexing='ij')
points = np.vstack([Xc.ravel(), Yc.ravel()]).T
u_ref_on_coarse = interp(points).reshape(Xc.shape)

# Figure 1 : solution coarse
plt.figure(figsize=(6,5))
plt.contourf(Xc, Yc, u, levels=40, cmap='viridis')
plt.colorbar(label='u')
plt.title('Solution 2D (coarse grid)')
plt.xlabel('x'); plt.ylabel('y')
plt.tight_layout()
plt.savefig('pde2d_solution.png', dpi=300)
plt.close()

# Figure 2 : L2 error map (pointwise absolute diff)
plt.figure(figsize=(6,5))
err_map = np.abs(u - u_ref_on_coarse)
plt.contourf(Xc, Yc, err_map, levels=40, cmap='inferno')
plt.colorbar(label='|u - u_ref|')
plt.title('Erreur absolue (coarse vs reference)')
plt.xlabel('x'); plt.ylabel('y')
plt.tight_layout()
plt.savefig('pde2d_L2error.png', dpi=300)
plt.close()

# Figure 3 : norm of gradient magnitude
ux = np.gradient(u, x, axis=0)
uy = np.gradient(u, y, axis=1)
gradnorm = np.sqrt(ux**2 + uy**2)
plt.figure(figsize=(6,5))
plt.contourf(Xc, Yc, gradnorm, levels=40, cmap='plasma')
plt.colorbar(label='|grad u|')
plt.title('Norme du gradient |\\nabla u|')
plt.xlabel('x'); plt.ylabel('y')
plt.tight_layout()
plt.savefig('pde2d_gradnorm.png', dpi=300)
plt.close()
plt.show()
# Print L2 global
L2_global = compute_L2_error(u, u_ref_on_coarse, x, y)
print(f"L2 global (coarse vs ref) = {L2_global:.3e}")
print("Figures sauvegardées : pde2d_solution.png, pde2d_L2error.png, pde2d_gradnorm.png")
