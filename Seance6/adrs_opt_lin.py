# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 11:12:36 2025

@author: SCD UM
"""

"""
optim_adrs.py

Optimisation ADRS en exploitant la linéarité par rapport aux contrôles.
- Construction de A_ij et B_i via interpolation sur un maillage de fond commun.
- Résolution linéaire (A x = B) et comparaison avec référence sur maillage fixe.
- Tracé de la surface J(x1,x2) en faisant varier les deux premiers contrôles.

Usage: python optim_adrs.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# ----------------------------
# Fonctions utilitaires
# ----------------------------
def phi_basis(x, ic, L=1.0):
    """Fonction de base spatiale pour le contrôle ic (ic=0..nbc-1)."""
    # centres L/(ic+1) comme dans ton code original
    center = L / (ic + 1)
    return np.exp(-100.0 * (x - center) ** 2)

def interp_to_bg(x_src, u_src, x_bg):
    """Interpoler u_src défini en x_src sur le maillage x_bg."""
    return np.interp(x_bg, x_src, u_src)

def trapz_integral(y, x):
    """Intégrale numérique par trapèze."""
    return np.trapz(y, x)

# ----------------------------
# Solveur ADRS 1D (temps-marquage -> steady)
# version simple, robuste; possibilité d'adaptation (itérations)
# retourne (x, T) et aussi la solution interpolée sur x_bg si fourni
# ----------------------------
def solve_adrs(NX, xcontrol, adapt=False, n_adapt=2, Time=10.0, K=0.01, V=1.0, lam=1.0,
               nb_bg=1000, L=1.0, verbose=False):
    """
    NX: nombre initial de points
    xcontrol: vecteur de coefficients de contrôle (taille nbc)
    adapt: si True, on effectue n_adapt itérations d'adaptation du maillage
    renvoie: (x_final, T_final, T_on_bg, x_bg)
    """
    nbc = len(xcontrol)
    # maillage de fond pour interpolation/assemblage (très fin pour précision)
    x_bg = np.linspace(0.0, L, nb_bg)

    # initialisation maillage
    x = np.linspace(0.0, L, NX)
    T = np.zeros_like(x)

    # Termes sources spatiaux (fixes) multipliés par coefficients de contrôle
    # F_total(x) = sum_i xcontrol[i] * phi_i(x)
    # mais dans le solveur on utilisera discrétisation
    # pour adaptation: itérations d'approximation basées sur Txx
    for adapt_it in range(max(1, n_adapt) if adapt else 1):
        # (re)initialise le champ T sur le maillage courant
        if adapt_it > 0:
            # raffinement simple basé sur métrique |Txx|
            # calcul Txx discret sur x (centré)
            NXcur = len(x)
            Txx = np.zeros_like(T)
            for j in range(1, NXcur - 1):
                dx_left = x[j] - x[j - 1]
                dx_right = x[j + 1] - x[j]
                # approx centrale non uniforme
                Txx[j] = ( (T[j+1] - T[j]) / dx_right - (T[j] - T[j-1]) / dx_left ) / (0.5*(dx_right+dx_left))
            # construire métrique m ~ |Txx| and convert to local h
            m = np.abs(Txx)
            m[0] = m[1]; m[-1] = m[-2]
            # normaliser m et construire hloc (on garde bornes)
            hmin = 0.005; hmax = 0.05
            mmax = m.max() if m.max()>0 else 1.0
            mnorm = m / mmax
            # densité de points : plus mnorm grand -> plus de points
            # nouvelle grille: construire cumulative
            Nnew = int(NXcur + np.round( (NXcur-1) * mnorm.mean() * 2 ))
            Nnew = max(Nnew, NXcur)  # ne pas diminuer ici
            x = np.linspace(0.0, L, Nnew)
            T = np.interp(x, np.linspace(0.0, L, NXcur), T)  # interpoler solution init
            if verbose:
                print(f"[adapt it {adapt_it}] nouveau NX = {len(x)}")

        # maintenant on solve en temps jusqu'à quasi-stationnaire
        # assemble source F on maillage x
        F = np.zeros_like(x)
        for ic in range(nbc):
            F += xcontrol[ic] * phi_basis(x, ic, L=L)

        # paramètres numériques
        dx = np.min(np.diff(x))
        # stable dt estimate (conservateur)
        dt = 0.25 * dx**2 / (V*dx + 4*K + 1e-12)
        t = 0.0
        T[:] = 0.0  # on commence de zéro
        max_steps = 100000
        # critère de convergence par résidu
        for step in range(max_steps):
            T_old = T.copy()
            # mise à jour explicite
            # second dérivé non uniforme approximé
            Tn = T.copy()
            for j in range(1, len(x)-1):
                dxl = x[j] - x[j-1]
                dxr = x[j+1] - x[j]
                # première dérivée centrée
                Tx = (Tn[j+1] - Tn[j-1]) / (dxl + dxr)
                # seconde dérivée non uniforme
                Txx = 2.0 * ( (Tn[j+1] - Tn[j]) / (dxr*(dxl+dxr)) - (Tn[j] - Tn[j-1]) / (dxl*(dxl+dxr)) )
                visc = K
                RHS = -V*Tx + visc*Txx - lam*Tn[j] + F[j]
                T[j] = Tn[j] + dt * RHS
            # conditions aux limites simples
            T[0] = 0.0
            T[-1] = T[-2]  # Neumann ~ 0
            t += dt
            # résidu
            res = np.linalg.norm(T - T_old, ord=1)
            if res < 1e-8 or t >= Time:
                break

        if verbose:
            print(f"solve_adrs: final NX={len(x)}, steps={step}, t={t:.3f}, res={res:.2e}")

    # interpolation sur maillage de fond
    T_on_bg = interp_to_bg(x, T, x_bg)
    return x, T, x_bg, T_on_bg

# ----------------------------
# Assemblage A, B via solutions de base (avec interpolation sur maillage de fond)
# ----------------------------
def assemble_A_B(nbc, NX_base, adapt=False, n_adapt=2, nb_bg=2000, verbose=False):
    """
    Calcule A (nbc x nbc) et B (nbc) en utilisant :
    - u0 : solution pour xcontrol=0
    - u_i : solution pour xcontrol = unit vector e_i
    Interpole tout sur maillage de fond x_bg (nb_bg points).
    """
    # 1) calculer u_des (cible) : ici pour test, on génère une cible via un choix xcible
    # mais on laisse la construction de u_des en dehors si nécessaire
    # Ici on simply will return A,B and the u_i on x_bg
    x0_control = np.zeros(nbc)
    # calcul u0
    _, u0, x_bg, u0_bg = solve_adrs(NX_base, x0_control, adapt=adapt, n_adapt=n_adapt, nb_bg=nb_bg, verbose=verbose)

    # calcul des u_i
    U_bg = np.zeros((nbc, nb_bg))
    for ic in range(nbc):
        ctrl = np.zeros(nbc)
        ctrl[ic] = 1.0
        _, u_i, _, u_i_bg = solve_adrs(NX_base, ctrl, adapt=adapt, n_adapt=n_adapt, nb_bg=nb_bg, verbose=verbose)
        U_bg[ic, :] = u_i_bg

    # assembler A,B requires u_des; here we only compute A and the basis U_bg and u0_bg
    A = np.zeros((nbc, nbc))
    for i in range(nbc):
        for j in range(i, nbc):
            A_ij = trapz_integral(U_bg[i, :] * U_bg[j, :], x_bg)
            A[i, j] = A_ij
            A[j, i] = A_ij
    return A, U_bg, u0_bg, x_bg

# ----------------------------
# Exemple d'utilisation et expérience demandée
# ----------------------------
def main_experiment():
    # paramètres utilisateur
    nbc = 6                # nb de contrôles
    NX_base = 30           # maille initiale pour les solves adaptifs
    adapt = True           # activer adaptation
    n_adapt = 2
    nb_bg = 2000           # maille de fond pour intégration (choix important pour précision)

    # 1) Construire une cible u_des en choisissant des contrôles "xcible"
    xcible = np.arange(1, nbc+1) * 0.5  # exemple, vecteur de cibles
    # calculer u_des via solve_adrs (avec adaptation ou pas selon ton choix)
    _, u_des, x_bg, u_des_bg = solve_adrs(NX_base, xcible, adapt=adapt, n_adapt=n_adapt, nb_bg=nb_bg, verbose=True)

    # 2) Construire base U_bg et A
    print("Assemblage des solutions de base (plus coûteux)...")
    t0 = time.time()
    A, U_bg, u0_bg, x_bg = assemble_A_B(nbc, NX_base, adapt=adapt, n_adapt=n_adapt, nb_bg=nb_bg, verbose=False)
    t1 = time.time()
    print(f"Assemble A and basis took {t1-t0:.1f}s")

    # 3) Calculer B = \int (u_des - u0) * u_i
    r_bg = u_des_bg - u0_bg
    B = np.zeros(nbc)
    for i in range(nbc):
        B[i] = trapz_integral(r_bg * U_bg[i, :], x_bg)

    # 4) Résolution linéaire A x = B
    x_opt_lin = np.linalg.solve(A, B)
    print("Solution linéaire x_opt_lin =", np.round(x_opt_lin, 4))

    # 5) Calculer J pour contrôle x (utilisant la reconstruction linéaire sur maillage de fond)
    def J_from_coeffs(xvec):
        # u = u0 + sum x_i u_i (sur maillage de fond)
        u_rec = u0_bg.copy()
        for i in range(len(xvec)):
            u_rec += xvec[i] * U_bg[i, :]
        diff = u_rec - u_des_bg
        return 0.5 * trapz_integral(diff * diff, x_bg)

    J_lin_opt = J_from_coeffs(x_opt_lin)
    print("J(x_opt_lin) =", J_lin_opt)

    # 6) Construction d'une référence sur un maillage fixe très fin (pour comparer)
    NX_ref = 200
    _, u_ref, x_bg_ref, u_ref_bg = solve_adrs(NX_ref, x_opt_lin, adapt=False, n_adapt=1, nb_bg=nb_bg, verbose=True)
    # calcul J_ref en comparant u_ref_bg à u_des_bg (déjà sur x_bg)
    # on interpole u_ref sur x_bg
    u_ref_on_bg = np.interp(x_bg, x_bg_ref, u_ref_bg)
    J_ref = 0.5 * trapz_integral((u_ref_on_bg - u_des_bg)**2, x_bg)
    print("J_ref (solution full solve on NX_ref) =", J_ref)

    # 7) Tracer la surface J(x1,x2) en balayant x1,x2
    # on fixe les autres contrôles à zéro
    idx1, idx2 = 0, 1
    ngrid = 41
    x1_vals = np.linspace(x_opt_lin[idx1] - 0.5, x_opt_lin[idx1] + 0.5, ngrid)
    x2_vals = np.linspace(x_opt_lin[idx2] - 0.5, x_opt_lin[idx2] + 0.5, ngrid)
    Jsurf = np.zeros((ngrid, ngrid))
    for i, a in enumerate(x1_vals):
        for j, b in enumerate(x2_vals):
            xtest = np.zeros(nbc)
            xtest[idx1] = a
            xtest[idx2] = b
            # reconstruct u using linear basis (fast)
            Jsurf[j, i] = J_from_coeffs(xtest)

    # affichage surface J
    plt.figure(figsize=(8,6))
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    cs = plt.contourf(X1, X2, Jsurf, 40, cmap='viridis')
    plt.colorbar(cs, label='J')
    plt.xlabel('x1'); plt.ylabel('x2')
    plt.title('Surface J(x1,x2) (autres contrôles fixés)')
    plt.scatter([x_opt_lin[idx1]],[x_opt_lin[idx2]], color='red', label='x_opt_lin')
    plt.legend()
    plt.show()

    # 8) Afficher comparaisons
    plt.figure()
    plt.plot(x_bg, u_des_bg, label='u_des (target)')
    plt.plot(x_bg, u0_bg, label='u0 (zero control)')
    # reconstruire u_opt_lin
    u_opt_lin = u0_bg.copy()
    for i in range(nbc):
        u_opt_lin += x_opt_lin[i] * U_bg[i, :]
    plt.plot(x_bg, u_opt_lin, '--', label='u_opt_lin (reconstruction)')
    plt.plot(x_bg, u_ref_on_bg, ':', label=f'u_ref on bg (NX_ref={NX_ref})')
    plt.legend(); plt.title('Comparaison solutions'); plt.xlabel('x'); plt.ylabel('T')
    plt.show()

    print("Fin de l'expérience.")

if __name__ == "__main__":
    main_experiment()
