# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 13:01:37 2025

@author: SCD UM
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# Paramètres physiques
K = 0.01      # Coefficient de diffusion
xmin = 0.0
xmax = 1.0
Time = 1.0    # Temps d'intégration
V = 1.0
lamda = 1.0
freq = 4 * math.pi  # Fréquence pour u(t) = sin(4*pi*t)

# Paramètres d'adaptation de maillage
niter_refinement = 10  # Nombre d'itérations d'adaptation
hmin = 0.01
hmax = 0.5
err = 0.01

# Paramètres numériques
NX = 10       # Nombre initial de points de grille
NT = 1000     # Nombre maximal de pas de temps
ifre = 100    # Fréquence de traçage
eps = 0.001   # Ratio de convergence relatif

# Fonction v(s) pour la solution exacte
def v_s(s):
    return np.exp(-20 * (s - (xmax + xmin) * 0.5) ** 2)

# Fonction u(t) pour la solution exacte
def u_t(t):
    return np.sin(freq * t)

# Dérivée temporelle de u(t)
def u_prime_t(t):
    return freq * np.cos(freq * t)

# Solution exacte u_ex(t, s) = u(t) * v(s)
def u_ex(t, s):
    return u_t(t) * v_s(s)

# Terme source f(s, t) = u'(t) * v(s) + V * u(t) * v'(s) - K * u(t) * v''(s) + lamda * u(t) * v(s)
def source_term(t, s):
    v_val = v_s(s)
    v_prime = -40 * (s - (xmax + xmin) * 0.5) * v_val
    v_second = (-40 + 1600 * (s - (xmax + xmin) * 0.5) ** 2) * v_val
    return u_prime_t(t) * v_val + V * u_t(t) * v_prime - K * u_t(t) * v_second + lamda * u_t(t) * v_val

# Boucle sur les stratégies de métrique (instationnaire ou stationnaire)
for metric_insta in [False, True]:
    errorL2 = []
    errorH1 = []
    itertab = []
    hloc = np.ones(NX) * hmax * 0.5
    itera = 0
    NX0 = 0

    while (np.abs(NX0 - NX) > 2 and itera < niter_refinement):
        itertab.append((xmax - xmin) / NX)
        itera += 1

        # Initialisation du maillage et de la solution
        x = np.linspace(xmin, xmax, NX)
        T = np.zeros(NX)

        # Adaptation du maillage
        if itera > 0:
            xnew = []
            Tnew = []
            nnew = 1
            xnew.append(xmin)
            Tnew.append(T[0])

            while xnew[nnew - 1] < xmax - hmin:
                for i in range(NX - 1):
                    if xnew[nnew - 1] >= x[i] and xnew[nnew - 1] <= x[i + 1] and xnew[nnew - 1] < xmax - hmin:
                        hll = (hloc[i] * (x[i + 1] - xnew[nnew - 1]) + hloc[i + 1] * (xnew[nnew - 1] - x[i])) / (x[i + 1] - x[i])
                        hll = min(max(hmin, hll), hmax)
                        nnew += 1
                        xnew.append(min(xmax, xnew[nnew - 2] + hll))
                        un = (T[i] * (x[i + 1] - xnew[nnew - 1]) + T[i + 1] * (xnew[nnew - 1] - x[i])) / (x[i + 1] - x[i])
                        Tnew.append(un)

            NX0 = NX
            NX = nnew
            x = np.zeros(NX)
            x[0:NX] = xnew[0:NX]
            T = np.zeros(NX)
            T[0:NX] = Tnew[0:NX]

        # Initialisation des variables
        rest = []
        RHS = np.zeros(NX)
        hloc = np.ones(NX) * hmax * 0.5
        metric = np.zeros(NX)

        # Calcul du pas de temps maximal
        dt = 1.e30
        for j in range(1, NX - 1):
            Tx = (v_s(x[j + 1]) - v_s(x[j - 1])) / (x[j + 1] - x[j - 1])
            Txip1 = (v_s(x[j + 1]) - v_s(x[j])) / (x[j + 1] - x[j])
            Txim1 = (v_s(x[j]) - v_s(x[j - 1])) / (x[j] - x[j - 1])
            Txx = (Txip1 - Txim1) / (0.5 * (x[j + 1] + x[j]) - 0.5 * (x[j] + x[j - 1]))
            dt = min(dt, 0.25 * (x[j + 1] - x[j - 1]) ** 2 / (V * np.abs(x[j + 1] - x[j - 1]) + 4 * K))

        print(f'NX = {NX}, Dt = {dt}')

        # Boucle en temps
        n = 0
        res = 1
        res0 = 1
        t = 0

        while n < NT and t < Time:
            n += 1
            dt = min(dt, Time - t)
            t += dt

            # Discrétisation de l'équation
            res = 0
            for j in range(1, NX - 1):
                visnum = 0.25 * (0.5 * (x[j + 1] + x[j]) - 0.5 * (x[j] + x[j - 1])) * np.abs(V)
                xnu = K + visnum
                Tx = (T[j + 1] - T[j - 1]) / (x[j + 1] - x[j - 1])
                Txip1 = (T[j + 1] - T[j]) / (x[j + 1] - x[j])
                Txim1 = (T[j] - T[j - 1]) / (x[j] - x[j - 1])
                Txx = (Txip1 - Txim1) / (0.5 * (x[j + 1] + x[j]) - 0.5 * (x[j] + x[j - 1]))
                src = source_term(t, x[j])
                RHS[j] = dt * (-V * Tx + xnu * Txx - lamda * T[j] + src)

                if metric_insta:
                    metric[j] += min(1. / hmin ** 2, max(1. / hmax ** 2, abs(Txx) / err))
                elif not metric_insta and (n == NT or t >= Time):
                    metric[j] = min(1. / hmin ** 2, max(1. / hmax ** 2, abs(Txx) / err))

                res += abs(RHS[j])

            # Conditions aux limites
            T[0] = 0
            T[NX - 1] = 2 * T[NX - 2] - T[NX - 3]

            # Mise à jour de la solution
            for j in range(1, NX - 1):
                T[j] += RHS[j]
                RHS[j] = 0

            if n == 1:
                res0 = res

            rest.append(res)

            # Visualisation à certains instants
            if n % ifre == 0 or t >= Time:
                plt.figure()
                plt.plot(x[0:NX], T[0:NX], label=f't = {t:.2f}', linestyle='--', marker='o')
                plt.xlabel('$x$', fontsize=12)
                plt.ylabel('$u(x,t)$', fontsize=12, rotation=0)
                plt.title('Solution instationnaire à différents instants')
                plt.legend()
                plt.grid(True)
                plt.savefig(f'solution_instationnaire_{t:.2f}.png', dpi=300, bbox_inches='tight')
                plt.close()

        # Moyenne des métriques en temps
        if metric_insta:
            metric[0:NX] /= n

        hloc[0:NX] = np.sqrt(1. / metric[0:NX])

        # Calcul des erreurs L2 et H1
        errL2h = 0
        errH1h = 0
        for j in range(1, NX - 1):
            Texx = (u_ex(t, x[j + 1]) - u_ex(t, x[j - 1])) / (x[j + 1] - x[j - 1])
            Tx = (T[j + 1] - T[j - 1]) / (x[j + 1] - x[j - 1])
            errL2h += (0.5 * (x[j + 1] + x[j]) - 0.5 * (x[j] + x[j - 1])) * (T[j] - u_ex(t, x[j])) ** 2
            errH1h += (0.5 * (x[j + 1] + x[j]) - 0.5 * (x[j] + x[j - 1])) * (Tx - Texx) ** 2

        errorL2.append(errL2h)
        errorH1.append(errL2h + errH1h)

        print(f'{metric_insta}, itera = {itera}, erreur L2, H1 = {errL2h}, {errH1h}')

        # Tracé du maillage adaptatif
        if metric_insta:
            plt.figure()
            plt.plot(x[0:NX], np.zeros_like(x[0:NX]), 'o', markersize=4, label='Points de maillage instationnaire')
            plt.xlabel('$x$', fontsize=12)
            plt.ylabel('Maillage adaptatif', fontsize=12, rotation=0)
            plt.title('Maillage instationnaire (moyenne temporelle des métriques)')
            plt.legend()
            plt.grid(True)
            plt.savefig('maillage_instationnaire.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.figure()
            plt.plot(x[0:NX], np.zeros_like(x[0:NX]), 'o', markersize=4, label='Points de maillage stationnaire')
            plt.xlabel('$x$', fontsize=12)
            plt.ylabel('Maillage adaptatif', fontsize=12, rotation=0)
            plt.title('Maillage adaptatif stationnaire (basé sur $t = T$)')
            plt.legend()
            plt.grid(True)
            plt.savefig('maillage_stationnaire.png', dpi=300, bbox_inches='tight')
            plt.close()

    # Tracé des erreurs
    plt.figure()
    plt.plot(itertab, errorL2, label='Erreur L2')
    plt.xlabel('Taille du maillage')
    plt.ylabel('Erreur L2')
    plt.legend()
    plt.grid(True)
    plt.savefig('erreur_L2.png', dpi=300, bbox_inches='tight')
    plt.close()
