# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 15:07:24 2026

@author: SCD UM
"""

import networkx as nx
import numpy as np

# Création d'un réseau simple (ex: 3x3)
G = nx.Graph()
nodes = [(i, j) for i in range(4) for j in range(4)]
G.add_nodes_from(nodes)

# Ajout des arêtes (connexions entre nœuds)
for i in range(3):
    for j in range(3):
        G.add_edge((i, j), (i + 1, j))  # Connexions horizontales
        G.add_edge((i, j), (i, j + 1))  # Connexions verticales

# Attribution de propriétés (ex: diamètre, longueur)
for u, v in G.edges():
    G.edges[u, v]['diameter'] = 5e-6  # 5 µm
    G.edges[u, v]['length'] = 50e-6   # 50 µm

# Calcul des conductivités hydrauliques (G = kA/μL)
k = 1e-10  # Perméabilité
mu = 1e-3  # Viscosité
for u, v in G.edges():
    d = G.edges[u, v]['diameter']
    L = G.edges[u, v]['length']
    A = np.pi * (d / 2)**2
    G.edges[u, v]['conductivity'] = (k * A) / (mu * L)

print("Réseau créé avec succès!")
print(f"Nombre de nœuds: {G.number_of_nodes()}")
print(f"Nombre d'arêtes: {G.number_of_edges()}")
