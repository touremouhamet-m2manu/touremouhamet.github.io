# adrs_insta_multiple_mesh_adap.py
import math
import numpy as np
import matplotlib.pyplot as plt
import sys

# ---------------------------
# Fonctions utilitaires
# ---------------------------

def v_spatial(x):
    """profil spatial v(s) utilisé pour construire u_ex(t,s)=u(t)*v(s)"""
    # un profil localisé (double gaussienne modulée)
    return 2*np.exp(-100*(x-0.25)**2) + np.exp(-200*(x-0.65)**2)

def build_source_from_exact(NX, x, time, V, K, lamda):
    """
    Calcule :
      - Tex (solution exacte u(t)*v(x)),
      - Text (d/dt Tex),
      - Texx (d/dx Tex) approximé,
      - F (terme source correspondant).
    u(t) = sin(4*pi*t), u'(t)=4*pi*cos(4*pi*t)
    """
    dx_local = np.zeros(NX)
    for j in range(1,NX):
        dx_local[j-1] = x[j]-x[j-1]
    dx = np.mean(dx_local) if NX>1 else 1.0

    Tex = np.zeros(NX)
    Text = np.zeros(NX)
    Texx = np.zeros(NX)
    F = np.zeros(NX)

    u_t = math.sin(4*math.pi*time)
    up_t = 4*math.pi*math.cos(4*math.pi*time)

    # valeur v(s) en nodal space
    v = v_spatial(x)

    for j in range(NX):
        Tex[j] = u_t * v[j]
        Text[j] = up_t * v[j]

    # dérivées spatiales discrètes centrales (intérieur)
    for j in range(1, NX-1):
        Texx[j] = (Tex[j+1] - Tex[j-1])/(x[j+1]-x[j-1])   # approx u_x
        Txx = (Tex[j+1] - 2*Tex[j] + Tex[j-1]) / ((0.5*(x[j+1]-x[j-1]))**2) # approx u_xx (approx)
        # Terme source garantissant que Tex est solution exacte
        F[j] = V*Texx[j] - K*Txx + lamda*Tex[j] + Text[j]

    # bords (prolongation / Neumann approximés)
    Texx[0] = Texx[1]
    Texx[-1] = Texx[-2]
    F[0] = F[1]
    F[-1] = F[-2]

    return F, Tex, Texx

def metric_from_Txx(NX, x, Txx_est, hmin, hmax, err_target):
    """
    Calcule une métrique (1/h^2) à partir d'une estimation du Txx (Hessien) :
    metric_j = clamp(|Txx|/err_target, 1/hmax^2, 1/hmin^2)
    puis lisse par moyenne, et renvoie hloc = sqrt(1/metric).
    """
    metric = np.zeros(NX)
    for j in range(1,NX-1):
        metric[j] = max(1.0/hmax**2, min(1.0/hmin**2, abs(Txx_est[j]) / max(err_target, 1e-16)))
    metric[0] = metric[1]
    metric[-1] = metric[-2]
    # lissage simple
    for j in range(0, NX-1):
        metric[j] = 0.5*(metric[j] + metric[j+1])
    metric[-1] = metric[-2]
    hloc = np.sqrt(1.0 / metric)
    # borne
    hloc = np.minimum(np.maximum(hloc, hmin), hmax)
    return hloc, metric

def build_mesh_from_hloc(xmin, xmax, hloc_nodes, x_nodes_of_hloc, hmin, hmax):
    """
    Construit un maillage xnew en parcourant [xmin,xmax] et en
    prenant un pas local h interpolé depuis hloc_nodes defined at x_nodes_of_hloc.
    """
    xnew = [xmin]
    Nloc = len(hloc_nodes)
    while xnew[-1] < xmax - 1e-12:
        curr = xnew[-1]
        # interpolation linéaire de hloc
        idx = np.searchsorted(x_nodes_of_hloc, curr) - 1
        if idx < 0:
            idx = 0
        if idx >= Nloc-1:
            idx = Nloc-2
        xL = x_nodes_of_hloc[idx]; xR = x_nodes_of_hloc[idx+1]
        w = 0.0 if (xR==xL) else (curr - xL)/(xR - xL)
        hloc_curr = (1-w)*hloc_nodes[idx] + w*hloc_nodes[idx+1]
        hloc_curr = min(max(hloc_curr, hmin), hmax)
        nxt = min(xmax, curr + hloc_curr)
        if nxt <= curr + 1e-12:
            nxt = min(xmax, curr + hmin)
        xnew.append(nxt)
        if len(xnew) > 200000:
            raise RuntimeError("Trop de points dans nouveau maillage (boucle).")
    return np.array(xnew)

def interp_to_background(x_source, T_source, background_mesh):
    """Interpolation linéaire de T_source défini sur x_source vers background_mesh."""
    TB = []
    for xb in background_mesh:
        if xb <= x_source[0]:
            TB.append(T_source[0])
        elif xb >= x_source[-1]:
            TB.append(T_source[-1])
        else:
            idx = np.searchsorted(x_source, xb) - 1
            if idx < 0: idx = 0
            if idx >= len(x_source)-1: idx = len(x_source)-2
            xL = x_source[idx]; xR = x_source[idx+1]
            w = (xb - xL)/(xR - xL)
            TB.append((1-w)*T_source[idx] + w*T_source[idx+1])
    return np.array(TB)

# ---------------------------
# Paramètres (physiques & adaptation)
# ---------------------------
K = 0.01
xmin = 0.0
xmax = 1.0
Time = 1.0   # on met Time=1s pour visualisations demandées
V = 1.0
lamda = 1.0

niter_refinement = 30
hmin = 0.01
hmax = 0.15
err_target = 0.03   # tol pour la métrique

# critères d'arrêt mixtes
tol_L2 = 1e-4       # tolérance L2 sur erreur (critère d'arrêt)
tol_NX_change = 2   # changement max toléré sur NX pour considérer mesh stable
min_iter = 2        # minimum d'itérations d'adaptation

# numériques temps
NX = 5
NT = 200000
eps = 1e-6

# fond pour interpolation et calcul Cauchy
NX_background = 400
background_mesh = np.linspace(xmin, xmax, NX_background)

# tableaux pour suivi
errorL2 = np.zeros(niter_refinement)
itertab = np.zeros(niter_refinement)
mesh_history = []

# flags
itera = 0
NX0 = 0

# Pour comparer adaptation stationnaire (basée sur Tex(t=Time))
do_stationary_control = True

# On stocke les solutions à instants demandés pour la visualisation
instants_to_plot = [0.0, 0.25*Time, 0.5*Time, Time]
solutions_at_instants = {t: None for t in instants_to_plot}
exact_at_instants = {t: None for t in instants_to_plot}

print("=== Lancement adaptation instationnaire (Time =", Time, "s) ===")

# ---------------------------
# Boucle d'adaptation principale
# ---------------------------
mesh_ok = False
error_ok = False

while (not (mesh_ok and error_ok) and itera < niter_refinement):
    itera += 1
    itertab[itera-1] = 1.0 / max(1, NX)
    print("\n--- Iteration d'adaptation", itera, " (NX=", NX, ") ---")
    mesh_history.append(NX)

    # initialisation du maillage / solution sur le maillage courant
    x = np.linspace(xmin, xmax, NX)
    T = np.zeros(NX)

    # variables pour accumuler métrique en temps (pour adaptation instationnaire)
    metric_time_sum = np.zeros(NX)
    metric_time_count = 0

    # variables pour enregistrement solution à instants demandés (on stocke sur background)
    sol_instants_bg = {t: None for t in instants_to_plot}
    exact_instants_bg = {t: None for t in instants_to_plot}

    # integration en temps sur le maillage courant
    t = 0.0
    n = 0
    rest = []
    res0 = 1.0

    # dt initial estimé (sûr)
    if NX>2:
        dx_est = np.min(np.diff(x))
        dt = 0.5 * dx_est**2 / (V*dx_est + 4*K + 1.0)
    else:
        dt = 1e-3

    while t < Time and n < NT:
        n += 1
        # recalcul du terme source exact pour l'instant courant
        F, Tex, Texx = build_source_from_exact(NX, x, t, V, K, lamda)

        # évaluation d'un dt stable (CFL-like)
        # on prend la plus petite cellule pour être conservatif
        dx_min = np.min(np.diff(x)) if NX>1 else 1.0
        dt = min(dt, 0.5*(dx_min**2) / (V*dx_min + 4*K + abs(np.max(F))*dx_min**2 + 1e-16))
        if dt <= 0:
            dt = 1e-6

        # calcul métrique locale basé sur estimation de Txx (ici on utilise Texx)
        # (on pourrait aussi utiliser T numérique, mais Texx guide adaptation vers la vérité)
        hloc_current, metric_current = metric_from_Txx(NX, x, Texx, hmin, hmax, err_target)

        # accumulateur de métriques en temps (intersection temporelle approchée par moyenne)
        # On interpole metric_current si la taille du maillage change dans le temps, mais ici mesh fixe
        metric_time_sum += metric_current
        metric_time_count += 1

        # schéma explicite (Euler-like) avec viscosité numérique (comme dans ton code)
        RHS = np.zeros(NX)
        res = 0.0
        for j in range(1, NX-1):
            visnum = 0.5*(0.5*(x[j+1]+x[j]) - 0.5*(x[j]+x[j-1]))*abs(V)
            xnu = K + visnum
            Tx = (T[j+1] - T[j-1]) / (x[j+1] - x[j-1])
            Txip1 = (T[j+1] - T[j])/(x[j+1] - x[j])
            Txim1 = (T[j] - T[j-1])/(x[j] - x[j-1])
            Txx_num = (Txip1 - Txim1) / (0.5*(x[j+1]+x[j]) - 0.5*(x[j]+x[j-1]))
            RHS[j] = dt * (-V*Tx + xnu*Txx_num - lamda*T[j] + F[j])
            res += abs(RHS[j])

        # mise à jour solution
        for j in range(1, NX-1):
            T[j] += RHS[j]
        T[-1] = T[-2]  # Neumann 0 en bord droit (approx)

        # sauvegarde solutions/tex à instants demandés (on prend condition time crossing)
        for tplot in instants_to_plot:
            if (solutions_at_instants[tplot] is None) and abs(t - tplot) <= dt/2:
                # on interpole T et Tex vers background pour comparer uniformément
                sol_instants_bg[tplot] = interp_to_background(x, T, background_mesh)
                exact_instants_bg[tplot] = interp_to_background(x, Tex, background_mesh)
                solutions_at_instants[tplot] = sol_instants_bg[tplot]
                exact_at_instants[tplot] = exact_instants_bg[tplot]

        # récupération du premier residual pour normalisation
        if n == 1:
            res0 = max(res, 1e-16)
        rest.append(res)

        t += dt

    # fin intégration en temps sur le maillage courant
    # calcul métrique moyenne en temps (intersection temporelle approchée par moyenne)
    if metric_time_count > 0:
        metric_time_avg = metric_time_sum / metric_time_count
    else:
        metric_time_avg = metric_current

    # on en déduit hloc moyen
    hloc_time_avg = np.sqrt(1.0 / np.maximum(metric_time_avg, 1e-16))
    hloc_time_avg = np.minimum(np.maximum(hloc_time_avg, hmin), hmax)

    # calcul erreur L2 finale (sur maillage courant en utilisant Tex à t=Time)
    # pour Tex final on ré-évalue au temps Time sur la grille x
    Ff, Tex_final, Texx_final = build_source_from_exact(NX, x, Time, V, K, lamda)
    # interpolation T et Tex_final sur cellules et calcul L2 approximé
    errL2h = 0.0
    for j in range(1, NX-1):
        celllen = 0.5*(x[j+1]+x[j]) - 0.5*(x[j]+x[j-1])
        # approx T (déjà sur noeuds) and Tex_final
        errL2h += celllen * (T[j] - Tex_final[j])**2
    errorL2[itera-1] = errL2h

    print("  itération", itera, "NX_current =", NX, " errL2 =", errL2h, " n_time_steps =", metric_time_count)

    # --- Critère mixte d'arrêt :
    # On veut que *les deux* conditions soient réalisées :
    #  1) stabilisation du nombre de noeuds (|NX - NX0| <= tol_NX_change)
    #  2) erreur L2 finale <= tol_L2
    mesh_ok = (abs(NX - NX0) <= tol_NX_change) and (itera >= min_iter)
    error_ok = (errL2h <= tol_L2) and (itera >= min_iter)

    # pour la sortie on impose de continuer tant que *les deux* ne sont pas vraies
    stop_now = mesh_ok and error_ok

    # OPTION : adaptation stationnaire basée sur la solution finale (Tex_final)
    if do_stationary_control:
        # calcule métrique stationnaire basée uniquement sur Tex_final (au temps Time)
        # estimation Txx_final (déjà en Texx_final)
        hloc_stationary, metric_stationary = metric_from_Txx(NX, x, Texx_final, hmin, hmax, err_target)
        # construit maillage "stationnaire" à partir de cette métrique
        x_stationary = build_mesh_from_hloc(xmin, xmax, hloc_stationary, x, hmin, hmax)
    else:
        x_stationary = None

    # --- construction du nouveau maillage à partir de la métrique moyenne en temps
    # on utilise x (coordonnées actuelles) comme référence pour hloc_time_avg nodes
    xnew = build_mesh_from_hloc(xmin, xmax, hloc_time_avg, x, hmin, hmax)
    nnew = len(xnew)

    # interpolation solution T sur xnew pour prochaine itération
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

    # mise à jour NX0/NX pour critère
    NX0 = NX
    NX = nnew

    # si on arrête à cause de tol (mais on veut double sécurité), on met stop flag
    if stop_now:
        print("Critère mixte réalisé: mesh stable ET erreur L2 tolérée -> arrêt.")
        break
    else:
        print("-> adaptation continue (mesh_ok=", mesh_ok, ", error_ok=", error_ok, ")")
        # préparer boucle : remplacer x,T
        x = xnew.copy()
        T = Tnew.copy()
        # sauvegarder maillage pour historique
        mesh_history.append(NX)

# ---------------------------
# Affichages et diagnostics
# ---------------------------

# 1) visualiser solutions exactes et numériques aux instants demandés (interpolées sur background_mesh)
plt.figure(figsize=(10,7))
for tplot in instants_to_plot:
    if solutions_at_instants.get(tplot) is not None:
        plt.plot(background_mesh, solutions_at_instants[tplot], label=f"Num t={tplot:.2f}s")
        plt.plot(background_mesh, exact_at_instants[tplot], '--', label=f"Exact t={tplot:.2f}s")
    else:
        print("Remarque: pas de sauvegarde de t=", tplot, " (pas de crossing exact dt)")

plt.xlabel("x"); plt.ylabel("T")
plt.title("Solutions numérique vs exacte à instants demandés")
plt.legend()
plt.grid(True)
plt.show()

# 2) afficher maillage adaptatif final et, si demandé, maillage stationnaire
plt.figure(figsize=(8,3))
# tracer noeuds du maillage final sur l'axe x
x_final = xnew
y_final = np.zeros_like(x_final)
plt.plot(x_final, y_final, 'o', label=f"Maillage adapt final (N={len(x_final)})")
if x_stationary is not None:
    y_stat = np.ones_like(x_stationary)*0.05
    plt.plot(x_stationary, y_stat, 'x', label=f"Maillage stationnaire (from Tex(t=Time), N={len(x_stationary)})")
plt.ylim([-0.05,0.1])
plt.title("Noeuds du maillage final et du maillage stationnaire")
plt.legend()
plt.grid(True)
plt.show()

# 3) courbe d'erreur L2 par itération d'adaptation
plt.figure()
it_range = np.arange(1, itera+1)
plt.plot(it_range, errorL2[:itera], 'o-')
plt.xlabel("Itération d'adaptation")
plt.ylabel("Erreur L2 (approx intégrée)")
plt.title("Erreur L2 par itération d'adaptation")
plt.grid(True)
plt.show()

# 4) afficher évolution du nombre de noeuds
plt.figure()
plt.plot(np.arange(1, len(mesh_history)+1), mesh_history, 's-')
plt.xlabel("Itération d'adaptation (historique)")
plt.ylabel("Nombre de noeuds")
plt.title("Evolution du nombre de noeuds lors de l'adaptation")
plt.grid(True)
plt.show()

print("=== Fin routine adaptation. itérations réalisées =", itera, " ===")
