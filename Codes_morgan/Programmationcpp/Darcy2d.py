# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 15:10:12 2026

@author: SCD UM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

class DarcySolver2D:
    def __init__(self, nx=10, ny=10, Lx=1.0, Ly=1.0, k=1e-12, mu=1e-3):
        """
        Initialise le solveur Darcy 2D.

        Paramètres:
        - nx, ny: Nombre de cellules dans les directions x et y.
        - Lx, Ly: Longueurs physiques du domaine (m).
        - k: Perméabilité du milieu (m²).
        - mu: Viscosité du fluide (Pa·s).
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.k = k
        self.mu = mu
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.n_nodes = (nx + 1) * (ny + 1)  # Nombre total de nœuds

        # Conditions aux limites (exemple: pression imposée aux bords gauche et droit)
        self.P_left = 1.0  # Pression à gauche (Pa)
        self.P_right = 0.0  # Pression à droite (Pa)
        self.P_top = None  # Pression en haut (None = condition de Neumann)
        self.P_bottom = None  # Pression en bas (None = condition de Neumann)

    def build_system(self):
        """
        Construit le système linéaire pour la loi de Darcy.
        Retourne la matrice A et le vecteur b pour le système Ax = b.
        """
        # Initialisation de la matrice creuse et du vecteur second membre
        A = lil_matrix((self.n_nodes, self.n_nodes))
        b = np.zeros(self.n_nodes)

        # Boucle sur tous les nœuds
        for i in range(self.nx + 1):
            for j in range(self.ny + 1):
                node = i * (self.ny + 1) + j

                # Conditions aux limites de Dirichlet (pression imposée)
                if i == 0:  # Bord gauche
                    A[node, node] = 1.0
                    b[node] = self.P_left
                    continue
                elif i == self.nx:  # Bord droit
                    A[node, node] = 1.0
                    b[node] = self.P_right
                    continue
                elif j == 0 and self.P_bottom is not None:  # Bord bas
                    A[node, node] = 1.0
                    b[node] = self.P_bottom
                    continue
                elif j == self.ny and self.P_top is not None:  # Bord haut
                    A[node, node] = 1.0
                    b[node] = self.P_top
                    continue

                # Équation de Darcy pour les nœuds internes
                if i > 0:  # Voisin gauche
                    left_node = (i - 1) * (self.ny + 1) + j
                    A[node, node] += (self.k * self.dy) / (self.mu * self.dx)
                    A[node, left_node] = - (self.k * self.dy) / (self.mu * self.dx)

                if i < self.nx:  # Voisin droit
                    right_node = (i + 1) * (self.ny + 1) + j
                    A[node, node] += (self.k * self.dy) / (self.mu * self.dx)
                    A[node, right_node] = - (self.k * self.dy) / (self.mu * self.dx)

                if j > 0:  # Voisin bas
                    bottom_node = i * (self.ny + 1) + (j - 1)
                    A[node, node] += (self.k * self.dx) / (self.mu * self.dy)
                    A[node, bottom_node] = - (self.k * self.dx) / (self.mu * self.dy)

                if j < self.ny:  # Voisin haut
                    top_node = i * (self.ny + 1) + (j + 1)
                    A[node, node] += (self.k * self.dx) / (self.mu * self.dy)
                    A[node, top_node] = - (self.k * self.dx) / (self.mu * self.dy)

        return A.tocsr(), b

    def solve(self):
        """
        Résout le système linéaire et retourne les pressions et vitesses.
        """
        A, b = self.build_system()
        P = spsolve(A, b)  # Résolution du système
        P = P.reshape((self.nx + 1, self.ny + 1))  # Remise en forme 2D

        # Calcul des vitesses (Darcy: v = -k/μ ∇P)
        U = np.zeros((self.nx, self.ny + 1))  # Vitesse en x (aux faces)
        V = np.zeros((self.nx + 1, self.ny))  # Vitesse en y (aux faces)

        # Vitesse en x (entre les nœuds)
        for i in range(self.nx):
            for j in range(self.ny + 1):
                U[i, j] = - (self.k / self.mu) * (P[i + 1, j] - P[i, j]) / self.dx

        # Vitesse en y (entre les nœuds)
        for i in range(self.nx + 1):
            for j in range(self.ny):
                V[i, j] = - (self.k / self.mu) * (P[i, j + 1] - P[i, j]) / self.dy

        return P, U, V

    def plot_results(self, P, U, V):
        """
        Affiche les résultats (pression et vitesses).
        """
        X = np.linspace(0, self.Lx, self.nx + 1)
        Y = np.linspace(0, self.Ly, self.ny + 1)

        plt.figure(figsize=(12, 5))

        # Tracé de la pression
        plt.subplot(1, 2, 1)
        plt.contourf(X, Y, P, levels=20, cmap='viridis')
        plt.colorbar(label='Pression (Pa)')
        plt.title('Champ de pression')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

        # Tracé des vitesses
        plt.subplot(1, 2, 2)
        X_u = np.linspace(0, self.Lx, self.nx)
        Y_u = np.linspace(0, self.Ly, self.ny + 1)
        X_v = np.linspace(0, self.Lx, self.nx + 1)
        Y_v = np.linspace(0, self.Ly, self.ny)
        plt.quiver(X_u, Y_u, U, np.zeros_like(U), scale=20, color='r', label='Vitesse en x')
        plt.quiver(X_v, Y_v, np.zeros_like(V), V, scale=20, color='b', label='Vitesse en y')
        plt.title('Champ de vitesse')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    solver = DarcySolver2D(nx=20, ny=20, Lx=0.1, Ly=0.1, k=1e-10, mu=1e-3)
    P, U, V = solver.solve()
    solver.plot_results(P, U, V)
