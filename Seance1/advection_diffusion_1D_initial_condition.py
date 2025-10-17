# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 17:30:24 2025

@author: SCD UM
"""

import numpy as np
import matplotlib.pyplot as plt

# Domain and BC
L = 1.0
Nx = 101
x = np.linspace(0, L, Nx)
u_left = 0.5          # u(0) = u_left (Dirichlet)
g_right = 0.0         # u_x(L) = g_right (Neumann)

# Construct a quadratic polynomial u0(x) = a x^2 + b x + c satisfying:
# u0(0) = c = u_left
# u0'(x) = 2 a x + b, so u0'(L) = 2 a L + b = g_right
# Fix also u0(mid)=value_mid for shape (choose value_mid)
value_mid = 1.0
# Solve for a,b,c:
c = u_left
# u0(L) = a L^2 + b L + c = value_right (we don't have value_right) -> use mid constraint
# Use constraints: u0(0)=c, u0(L) derivative, and u0(L/2)=value_mid
A = np.array([[ (L/2)**2, (L/2) ],
              [ 2*L, 1.0 ]])
rhs = np.array([ value_mid - c,
                 g_right - 0.0 ])  # second eq: 2aL + b = g_right
sol = np.linalg.solve(A, rhs)
a = sol[0]
b = sol[1]

u0 = a*x**2 + b*x + c

plt.figure(figsize=(8,4))
plt.plot(x, u0, '-b', label='u0(x)')
plt.plot([0], [u_left], 'ro', label='u(0)=u_left')
# numerical derivative at right
dx = x[1]-x[0]
u_x_right = (u0[-1]-u0[-2])/dx
plt.plot([L], [u0[-1]], 'go', label=f'u_x(L)≈{u_x_right:.3f}')
plt.title('Fonction initiale u0(x) compatible BC (exemple quadratique)')
plt.xlabel('x'); plt.ylabel('u0(x)')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig('u0_check.png', dpi=300)
plt.close()

print("Figure sauvegardée : u0_check.png")
