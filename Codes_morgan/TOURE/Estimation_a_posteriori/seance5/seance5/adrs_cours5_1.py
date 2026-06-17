import math
import numpy as np
import matplotlib.pyplot as plt
import sys

def adrs_fct(n, x, xmin, xmax):
    """
    Calcule une solution exacte / terme source d'exemple (Tex) et,
    si souhaité, un terme source F analytique.
    Ici Tex est une somme de deux pics gaussiens (comme dans votre code).
    Retour : Tex (taille n)
    """
    Tex = np.zeros(n)
    for j in range(n):
        Tex[j] = 2*np.exp(-100*(x[j]-(xmax+xmin)*0.25)**2) + np.exp(-200*(x[j]-(xmax+xmin)*0.65)**2)
    return Tex

def metric_fct(n, x, u, err, hmin, hmax):
    """
    Calcule une métrique locale à partir d'une estimation du 'Hessien' (ici
    discrétisation 1D) de u. On renvoie hloc (taille n) et metric (1/h^2).
    - u: vecteur nodal de la fonction (par ex la solution ou Tex)
    - x: coordonnées nodales
    - err: tolérance d'erreur souhaitée (contrôle de la métrique)
    """
    metric = np.zeros(n)
    # calcul discrétisé d'une estimation du second ordre (Txx)
    for j in range(1, n-1):
        # approximation Txx centrée adaptée à maillage non uniforme
        dx_plus = x[j+1] - x[j]
        dx_minus = x[j] - x[j-1]
        # formule adaptée au maillage non uniforme (approx second derivative)
        a = 2.0 / (dx_minus * (dx_minus + dx_plus))
        b = -2.0 / (dx_minus * dx_plus)
        c = 2.0 / (dx_plus * (dx_minus + dx_plus))
        Txx = a * u[j-1] + b * u[j] + c * u[j+1]
        metric[j] = max(1.0 / hmax**2, min(1.0 / hmin**2, abs(Txx) / max(err, 1e-16)))
    # bords : prolongation
    metric[0] = metric[1]
    metric[-1] = metric[-2]
    # lissage simple (moyenne)
    for j in range(0, n-1):
        metric[j] = 0.5*(metric[j] + metric[j+1])
    metric[-1] = metric[-2]
    hloc = np.sqrt(1.0 / metric)
    # s'assurer bornes
    hloc = np.minimum(np.maximum(hloc, hmin), hmax)
    return hloc, metric

def mesh_fct(xmin, xmax, hloc, hmin, hmax):
    """
    Construit un nouveau maillage xnew à partir de hloc (taille Nprev).
    Principe : on parcourt l'intervalle en ajoutant des pas = hloc_interpolé.
    Retour : xnew (liste)
    """
    xnew = [xmin]
    # interpolation continue de hloc sur l'espace courant
    # pour produire un pas local, on utilise une interpolation linéaire de hloc en fonction de x
    # on crée d'abord une fonction d'interpolation simple
    Nloc = len(hloc)
    x_nodes = np.linspace(xmin, xmax, Nloc)
    while xnew[-1] < xmax - 1e-12:
        curr = xnew[-1]
        # trouver position dans x_nodes pour interp de h
        idx = np.searchsorted(x_nodes, curr) - 1
        if idx < 0:
            idx = 0
        if idx >= Nloc-1:
            idx = Nloc-2
        xL = x_nodes[idx]; xR = x_nodes[idx+1]
        w = 0.0 if (xR == xL) else (curr - xL)/(xR - xL)
        hloc_curr = (1-w)*hloc[idx] + w*hloc[idx+1]
        hloc_curr = min(max(hloc_curr, hmin), hmax)
        nxt = min(xmax, curr + hloc_curr)
        # protéger contre stagnation numérique
        if nxt <= curr + 1e-12:
            nxt = curr + hmin
            if nxt > xmax:
                nxt = xmax
        xnew.append(nxt)
        if len(xnew) > 1000000:
            raise RuntimeError("Trop de points dans le nouveau maillage (boucle infinie).")
    return np.array(xnew)

# -------------------------
# Paramètres physiques / num
# -------------------------
iplot=0

K = 0.01     # Diffusion coefficient
xmin = 0.0
xmax = 1.0
Time = 10.0  # Integration time

V = 1.0
lamda = 1.0

# mesh adaptation param
niter_refinement = 30      # niter different calculations
hmin = 0.01
hmax = 0.15
err = 0.03

# NUMERICAL PARAMETERS
NX = 3    # Number of grid points : initialization
NT = 100000   # Number of time steps max
ifre = 1000000  # plot every ifre time iterations
eps = 0.001     # relative convergence ratio

errorL2 = np.zeros((niter_refinement))
errorH1 = np.zeros((niter_refinement))
itertab = np.zeros((niter_refinement))
hloc = np.ones((NX))*hmax

NX_background = 200
background_mesh = np.linspace(xmin, xmax, NX_background)
Tbacknew = []

itera = 0
NX0 = 0

# initial mesh (3 points)
x = np.linspace(xmin, xmax, NX)
T = np.zeros(NX)

while (abs(NX0 - NX) > 2 and itera < niter_refinement-1) or itera == 0:
    itera += 1
    itertab[itera] = 1.0 / max(1, NX)

    iplot = itera-2

    # si on est en début d'itération, on a déjà x,T; sinon le bloc adaptation mettra à jour
    # Calcul Tex (solution exacte / target) et éventuellement F
    Tex = adrs_fct(NX, x, xmin, xmax)

    # Estimation du pas de temps dt CFL-like basé sur le maillage courant
    dt = 1.e30
    for j in range(1, NX-1):
        Tx = (Tex[j+1] - Tex[j-1])/(x[j+1] - x[j-1])
        Txip1 = (Tex[j+1] - Tex[j])/(x[j+1] - x[j])
        Txim1 = (Tex[j] - Tex[j-1])/(x[j] - x[j-1])
        Txx = (Txip1 - Txim1)/(0.5*(x[j+1]+x[j]) - 0.5*(x[j]+x[j-1]))
        Ftemp = V*Tx - K*Txx + lamda*Tex[j]
        dt = min(dt, 0.5*(x[j+1] - x[j-1])**2/(V*abs(x[j+1] - x[j-1]) + 4*K + abs(Ftemp)*(x[j+1] - x[j-1])**2))

    print('iter global', itera, 'NX=', NX, 'Dt=', dt)

    # time stepping to compute approximate solution T on mesh x
    # --- initial condition: T is zero already
    n = 0
    res = 1.0
    res0 = 1.0
    t = 0.0
    rest = []

    # pre-allocation
    RHS = np.zeros(NX)
    metric = np.ones(NX)
    hloc = np.ones(NX) * (hmax*0.5)

    # compute forcing F from Tex to drive the problem (same as before)
    F = np.zeros(NX)
    for j in range(1, NX-1):
        Tx = (Tex[j+1] - Tex[j-1])/(x[j+1] - x[j-1])
        Txip1 = (Tex[j+1] - Tex[j])/(x[j+1] - x[j])
        Txim1 = (Tex[j] - Tex[j-1])/(x[j] - x[j-1])
        Txx = (Txip1 - Txim1)/(0.5*(x[j+1]+x[j]) - 0.5*(x[j]+x[j-1]))
        F[j] = V*Tx - K*Txx + lamda*Tex[j]

    # boucle de temps (simple Euler explicite incrémental tel que dans votre code)
    while (n < NT and (res/res0 > eps or n == 0) and t < Time):
        n += 1
        t += dt
        res = 0.0
        for j in range(1, NX-1):
            visnum = 0.5*(0.5*(x[j+1] + x[j]) - 0.5*(x[j] + x[j-1]))*abs(V)
            xnu = K + visnum
            Tx = (T[j+1] - T[j-1])/(x[j+1] - x[j-1])
            Txip1 = (T[j+1] - T[j])/(x[j+1] - x[j])
            Txim1 = (T[j] - T[j-1])/(x[j] - x[j-1])
            Txx = (Txip1 - Txim1)/(0.5*(x[j+1]+x[j]) - 0.5*(x[j]+x[j-1]))
            RHS[j] = dt * (-V*Tx + xnu*Txx - lamda*T[j] + F[j])
            # métrique locale provisoire basée sur Txx (sera recalc par metric_fct)
            metric[j] = min(1.0/hmin**2, max(1.0/hmax**2, abs(Txx)/err))
            res += abs(RHS[j])

        metric[0] = metric[1]
        metric[-1] = metric[-2]
        for j in range(0, NX-1):
            metric[j] = 0.5*(metric[j] + metric[j+1])
        metric[-1] = metric[-2]
        hloc = np.sqrt(1.0 / metric)
        # borne hloc
        hloc = np.minimum(np.maximum(hloc, hmin), hmax)

        for j in range(1, NX-1):
            T[j] += RHS[j]
            RHS[j] = 0.0
        # condition au bord (Neumann 0)
        T[-1] = T[-2]

        if n == 1:
            res0 = max(res, 1e-16)
        rest.append(res)

        if (n % ifre == 0) or (res/res0 < eps):
            print('  time-iter=', n, 'residual=', res)

    # interpolation de la solution sur le maillage de fond
    Tbackold = np.array(Tbacknew) if len(Tbacknew) else np.array([])
    Tbacknew = []
    # on interpole linéairement T(x) sur background_mesh
    for xb in background_mesh:
        # trouver interval
        if xb <= x[0]:
            Tbacknew.append(T[0])
        elif xb >= x[-1]:
            Tbacknew.append(T[-1])
        else:
            idx = np.searchsorted(x, xb) - 1
            if idx < 0: idx = 0
            if idx >= len(x)-1: idx = len(x)-2
            xL = x[idx]; xR = x[idx+1]
            w = (xb - xL)/(xR - xL)
            Tbacknew.append((1-w)*T[idx] + w*T[idx+1])
    Tbacknew = np.array(Tbacknew)

    if Tbackold.size == Tbacknew.size and Tbackold.size>0:
        cauchy = np.sum(np.abs(Tbacknew - Tbackold))
        print("  Cauchy norm on background mesh = ", cauchy)

    # Calcul d'erreurs L2/H1 approchées
    errL2h = 0.0
    errH1h = 0.0
    for j in range(1, NX-1):
        Texx = (Tex[j+1] - Tex[j-1])/(x[j+1] - x[j-1])
        Tx = (T[j+1] - T[j-1])/(x[j+1] - x[j-1])
        celllen = (0.5*(x[j+1]+x[j]) - 0.5*(x[j]+x[j-1]))
        errL2h += celllen * (T[j] - Tex[j])**2
        errH1h += celllen * (Tx - Texx)**2

    errorL2[itera] = errL2h
    errorH1[itera] = errL2h + errH1h
    print('  norm error L2, H1=', errL2h, errH1h)
    print('----------------------------------')

    # --- mesh adaptation étape : calcul métrique plus robuste à partir de Tex (ou T)
    # On calcule hloc à partir de Tex (pour guider adaptation vers la "vérité")
    hloc_calc, metric_calc = metric_fct(NX, x, Tex, err, hmin, hmax)

    # génère un nouveau maillage xnew à partir de hloc_calc
    xnew = mesh_fct(xmin, xmax, hloc_calc, hmin, hmax)
    # Interpoler T sur le nouveau maillage (initialisation pour prochaine itération)
    nnew = len(xnew)
    Tnew = np.zeros(nnew)
    for i in range(nnew):
        xi = xnew[i]
        if xi <= x[0]:
            Tnew[i] = T[0]
        elif xi >= x[-1]:
            Tnew[i] = T[-1]
        else:
            idx = np.searchsorted(x, xi) - 1
            if idx < 0: idx = 0
            if idx >= len(x)-1: idx = len(x)-2
            xL = x[idx]; xR = x[idx+1]
            w = (xi - xL)/(xR - xL)
            Tnew[i] = (1-w)*T[idx] + w*T[idx+1]

    NX0 = NX
    NX = nnew
    x = xnew.copy()
    T = Tnew.copy()

# Fin de la boucle d'adaptation
# affichages finaux
plt.figure()
plt.plot(background_mesh, Tbacknew, label='T (interpolé sur fond)')
# tracer la dernière Tex interpolée sur background pour comparaison
Tex_bg = np.array([adrs_fct(2, np.array([xb,xb]), xmin, xmax)[0] for xb in background_mesh]) # hack: adrs_fct renvoie un tableau; on reprend valeur
plt.plot(background_mesh, Tex_bg, label='Tex (approx)', linestyle='--')
plt.legend()
plt.title('Solution adaptative (sur maillage fond)')
plt.show()

# curve erreurs
valid_range = range(1, itera+1)
plt.figure()
plt.plot(itertab[1:itera+1], np.log10(errorL2[1:itera+1]), label='log10 L2')
plt.plot(itertab[1:itera+1], np.log10(errorH1[1:itera+1]), label='log10 H1')
plt.xlabel('1/N (approx)')
plt.legend()
plt.show()

# scatter NB pts vs Error
plt.figure()
plt.plot(1/itertab[1:itera+1], errorL2[1:itera+1], marker='o')
plt.xlabel('Nb points (approx)')
plt.ylabel('Error L2')
plt.title('NB pts vs Error L2')
plt.show()
