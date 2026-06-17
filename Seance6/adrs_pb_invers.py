# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 11:51:07 2025

@author: SCD UM
"""

"""
inverse_refinement_experiment.py

Expérience : inversion linéaire avec assemblage A x = b et boucle de raffinement.
- construit u_des = u(Xopt)
- calcule u_i (solutions pour e_i)
- assemble A,B sur un maillage de fond fin (interpolation)
- résout A x = b (linéaire)
- boucle sur différentes tailles de maille / raffinements pour observer convergence
- teste aussi le cas u_des = 1 (contrôle inconnu)
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------
# Dépendances : solve_adrs et utilitaires
# On inclut une version compacte et robuste du solveur explicite 1D
# ----------------------------
def phi_basis(x, ic, L=1.0):
    center = L / (ic + 1)
    return np.exp(-100.0 * (x - center) ** 2)

def interp_to_bg(x_src, u_src, x_bg):
    return np.interp(x_bg, x_src, u_src)

def trapz_integral(y, x):
    return np.trapz(y, x)

def solve_adrs(NX, xcontrol, adapt=False, n_adapt=1, Time=5.0, K=0.01, V=1.0, lam=1.0,
               nb_bg=2000, L=1.0, verbose=False):
    """
    Solveur ADR simplifié (explicite) renvoyant (x, T, x_bg, T_on_bg)
    - NX : nombre initial de points
    - xcontrol : vecteur de contrôles
    - adapt : si True, fait une itération de raffinement simple basée sur Txx
    - nb_bg : nombre de points du maillage de fond pour interpolation
    """
    nbc = len(xcontrol)
    x_bg = np.linspace(0.0, L, nb_bg)

    # maillage initial
    x = np.linspace(0.0, L, NX)
    T = np.zeros_like(x)

    # option d'adaptation simple (itérations)
    n_adapt = max(1, n_adapt) if adapt else 1
    for it in range(n_adapt):
        # définir F sur maillage courant
        F = np.zeros_like(x)
        for ic in range(nbc):
            F += xcontrol[ic] * phi_basis(x, ic, L=L)

        dx = np.min(np.diff(x))
        dt = 0.25 * dx**2 / (V * dx + 4 * K + 1e-12)
        t = 0.0

        # pas de temps explicite jusqu'à Time (ou convergence explicite)
        T[:] = 0.0
        max_steps = int(1e5)
        for step in range(max_steps):
            T_old = T.copy()
            Tn = T.copy()
            # mise à jour
            for j in range(1, len(x)-1):
                dxl = x[j] - x[j-1]
                dxr = x[j+1] - x[j]
                # dérivée première centrée (non uniforme approx)
                Tx = (Tn[j+1] - Tn[j-1]) / (dxl + dxr)
                # dérivée seconde non uniforme (approx stable)
                Txx = 2.0 * ( (Tn[j+1] - Tn[j]) / (dxr*(dxl+dxr)) - (Tn[j] - Tn[j-1]) / (dxl*(dxl+dxr)) )
                RHS = -V*Tx + K * Txx - lam * Tn[j] + F[j]
                T[j] = Tn[j] + dt * RHS
            # BC
            T[0] = 0.0
            T[-1] = T[-2]
            t += dt
            res = np.linalg.norm(T - T_old, ord=1)
            if res < 1e-8 or t >= Time:
                break

        if verbose:
            print(f"solve_adrs: NX={len(x)}, steps={step}, t={t:.3f}, res={res:.2e}")

        # adaptation grossière : raffiner en fonction de |Txx|
        if adapt and it < n_adapt - 1:
            Txx_arr = np.zeros_like(T)
            for j in range(1, len(x)-1):
                dxl = x[j] - x[j-1]
                dxr = x[j+1] - x[j]
                Txx_arr[j] = 2.0 * ( (T[j+1] - T[j]) / (dxr*(dxl+dxr)) - (T[j] - T[j-1]) / (dxl*(dxl+dxr)) )
            m = np.abs(Txx_arr)
            m[0] = m[1]; m[-1] = m[-2]
            # normalisation et construction d'un nouveau maillage proportionnel à m
            mnorm = m / (m.max() + 1e-12)
            # calculer nombre de points supplémentaires proportionnel à la moyenne de mnorm
            NXcur = len(x)
            add = int(np.round((NXcur-1) * mnorm.mean() * 2.0))
            Nnew = max(NXcur, NXcur + add)
            x = np.linspace(0.0, L, Nnew)
            T = np.interp(x, np.linspace(0.0, L, NXcur), T)

    # interpolation sur maillage de fond
    T_on_bg = interp_to_bg(x, T, x_bg)
    return x, T, x_bg, T_on_bg

# ----------------------------
# Fonction qui, pour une taille NX donnée, calcule A,B et X*
# ----------------------------
def compute_A_B_and_solve(NX, nbc, xcible=None, adapt=False, n_adapt=1, nb_bg=2000, L=1.0, verbose=False):
    """
    - Si xcible fourni : on construit u_des = u(xcible) (target)
    - calcule u0 (zero controls) et u_i (unit vectors)
    - assemble A,B sur maillage de fond x_bg
    - résout A x = B
    Retour : A, B, x_opt, u_des_bg, u0_bg, U_bg (nbc x nb_bg), x_bg
    """
    # 1) target u_des
    if xcible is None:
        raise ValueError("xcible must be provided to build u_des in this function")
    # calculer u_des
    _, u_des, x_bg, u_des_bg = solve_adrs(NX, xcible, adapt=adapt, n_adapt=n_adapt, nb_bg=nb_bg, L=L, verbose=verbose)

    # 2) calcule u0 (zero) et u_i
    _, u0, _, u0_bg = solve_adrs(NX, np.zeros(nbc), adapt=adapt, n_adapt=n_adapt, nb_bg=nb_bg, L=L, verbose=verbose)
    U_bg = np.zeros((nbc, len(x_bg)))
    for ic in range(nbc):
        ctrl = np.zeros(nbc)
        ctrl[ic] = 1.0
        _, ui, _, ui_bg = solve_adrs(NX, ctrl, adapt=adapt, n_adapt=n_adapt, nb_bg=nb_bg, L=L, verbose=verbose)
        U_bg[ic, :] = ui_bg

    # 3) assemble A and B
    A = np.zeros((nbc, nbc))
    for i in range(nbc):
        for j in range(i, nbc):
            Aij = trapz_integral(U_bg[i, :] * U_bg[j, :], x_bg)
            A[i, j] = Aij
            A[j, i] = Aij

    r_bg = u_des_bg - u0_bg
    B = np.zeros(nbc)
    for i in range(nbc):
        B[i] = trapz_integral(r_bg * U_bg[i, :], x_bg)

    # 4) solve
    x_opt = np.linalg.solve(A, B)

    # compute cost J
    u_rec = u0_bg.copy()
    for i in range(nbc):
        u_rec += x_opt[i] * U_bg[i, :]
    J = 0.5 * trapz_integral((u_rec - u_des_bg)**2, x_bg)

    return A, B, x_opt, J, u_des_bg, u0_bg, U_bg, x_bg

# ----------------------------
# Boucle de raffinement et tracés demandés
# ----------------------------
def experiment_refinement():
    nbc = 4  # nombre de contrôles demandés dans ton algorithme (Xopt en R^4)
    # définir Xopt (exemple donné)
    Xopt = np.array([1.0, 2.0, 3.0, 4.0])  # vecteur de référence
    # paramètres d'expérience
    NX_list = [30, 50, 80, 120]   # mailles successives (raffinement)
    adapt_flag = False
    n_adapt = 1
    nb_bg = 2000   # maillage de fond pour intégration

    # stockage résultats
    Xstar_list = []
    J_list = []
    errX_list = []
    hx_list = []

    print("=== Expérience : u_des = u(Xopt) ===")
    for NX in NX_list:
        print(f"\n--- NX = {NX} ---")
        # construire xcible = Xopt but must match nbc length
        xcible = Xopt.copy()
        # obtenir A,B,xstar
        A, B, xstar, J, udes_bg, u0_bg, U_bg, x_bg = compute_A_B_and_solve(NX, nbc, xcible,
                                                                           adapt=adapt_flag, n_adapt=n_adapt, nb_bg=nb_bg, verbose=False)
        Xstar_list.append(xstar)
        J_list.append(J)
        err = np.linalg.norm(xstar - Xopt)
        errX_list.append(err)
        hx_list.append(1.0/(NX-1))
        print("xstar =", xstar)
        print("J =", J, "errX =", err)

    Xstar_arr = np.array(Xstar_list)
    # tracés : chaque composante vs h
    plt.figure(figsize=(8,6))
    for k in range(nbc):
        plt.plot(hx_list, Xstar_arr[:, k], '-o', label=f'X*_{k+1}')
        plt.axhline(Xopt[k], color='k', linestyle='--', alpha=0.3)
    plt.gca().invert_xaxis()
    plt.xlabel('h (1/(NX-1))')
    plt.ylabel('Composantes X*(h)')
    plt.title('Composantes de X* en fonction du pas h (raffinement)')
    plt.legend(); plt.grid(True)
    plt.show()

    # J(h) et erreur ||X*-Xopt||
    plt.figure(figsize=(8,5))
    plt.subplot(1,2,1)
    plt.plot(hx_list, J_list, '-o')
    plt.gca().invert_xaxis()
    plt.xlabel('h'); plt.ylabel('J(X*)'); plt.title('J(X*(h))')
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(hx_list, errX_list, '-o')
    plt.gca().invert_xaxis()
    plt.xlabel('h'); plt.ylabel('||X*-Xopt||'); plt.title('Erreur sur X*')
    plt.grid(True)
    plt.show()

    # ----------------------------
    # Cas u_des = 1 (on ne connait pas Xopt)
    # On résout le même problème mais u_des = 1 sur tout le domaine (constante)
    # ----------------------------
    print("\n=== Expérience : u_des = 1 (contrôle inconnu) ===")
    Xstar_list2 = []
    J_list2 = []
    for NX in NX_list:
        # construire u_des = 1 on background by simple array
        # but compute A and U_bg first (we can reuse compute_A_B_and_solve by building an artificial u_des)
        # Build U_bg and u0_bg for this NX
        nbc_local = nbc
        # compute basis only (we reuse solve_adrs with xcible=zero for u0 and unit vectors)
        # we'll use compute_A_B_and_solve but trick: provide xcible equal zero then override u_des
        # compute basis
        # use compute_A_B_and_solve with a dummy xcible to get U_bg and u0_bg:
        dummy_xcible = np.zeros(nbc_local)
        A_tmp, B_tmp, xstar_tmp, J_tmp, udes_bg_tmp, u0_bg, U_bg, x_bg = compute_A_B_and_solve(
            NX, nbc_local, dummy_xcible, adapt=adapt_flag, n_adapt=n_adapt, nb_bg=nb_bg, verbose=False)
        # define u_des_bg as ones
        u_des_bg_const = np.ones_like(x_bg)
        # assemble B with u_des = 1
        B_const = np.zeros(nbc_local)
        for i in range(nbc_local):
            B_const[i] = trapz_integral(u_des_bg_const * U_bg[i, :], x_bg) - trapz_integral(u0_bg * U_bg[i, :], x_bg)
        # Solve A x = B_const
        xstar_const = np.linalg.solve(A_tmp, B_const)
        # compute J (with reconstruction)
        u_rec = u0_bg.copy()
        for i in range(nbc_local):
            u_rec += xstar_const[i] * U_bg[i, :]
        Jc = 0.5 * trapz_integral((u_rec - u_des_bg_const)**2, x_bg)
        Xstar_list2.append(xstar_const)
        J_list2.append(Jc)
        print(f"NX={NX}: xstar_const = {xstar_const}, J = {Jc}")

    Xstar_arr2 = np.array(Xstar_list2)
    # plots
    plt.figure(figsize=(8,6))
    for k in range(nbc):
        plt.plot(hx_list, Xstar_arr2[:, k], '-o', label=f'X*_{k+1}')
    plt.gca().invert_xaxis()
    plt.xlabel('h'); plt.ylabel('Composantes X*(h)'); plt.title('X*(h) pour u_des = 1')
    plt.legend(); plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(hx_list, J_list2, '-o')
    plt.gca().invert_xaxis()
    plt.xlabel('h'); plt.ylabel('J'); plt.title('J(X*(h)) pour u_des = 1')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    experiment_refinement()
