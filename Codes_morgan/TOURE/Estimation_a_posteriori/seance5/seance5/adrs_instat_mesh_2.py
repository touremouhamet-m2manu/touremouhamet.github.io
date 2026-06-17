import math
import numpy as np
import matplotlib.pyplot as plt

# PHYSICAL PARAMETERS
K = 0.01      # Diffusion coefficient
xmin = 0.0
xmax = 1.0
Time = 1.0    # Temps total d'intégration
V = 1.0
lamda = 1.0
freq = 7.0    # Fréquence pour le terme source

# MESH ADAPTATION PARAMETERS
niter_refinement = 10      # Nombre maximal d'itérations d'adaptation
hmin = 0.01
hmax = 0.5
err = 0.01

# NUMERICAL PARAMETERS
NX = 3                  # Nombre initial de points de maillage
NT = 10000              # Nombre maximal de pas de temps
ifre = 100000           # Fréquence de traçage
eps = 0.001             # Ratio de convergence relatif

# CRITÈRES D'ARRÊT MIXTES
NX_min = 50             # Seuil minimal pour le nombre de points de maillage
epsilon_L2 = 0.01       # Seuil pour l'erreur L2

# Initialisation des tableaux pour stocker les erreurs et itérations
errorL2 = np.zeros((niter_refinement))
itertab = np.zeros((niter_refinement))
hloc = np.ones((NX)) * hmax * 0.5
iter = 0
NX0 = 0

# Boucle d'adaptation du maillage
while (iter < niter_refinement):
    itertab[iter] = 1. / NX
    iter += 1

    # Initialisation du maillage
    x = np.linspace(xmin, xmax, NX)
    T = np.zeros((NX))

    # Adaptation du maillage (si ce n'est pas la première itération)
    if (iter > 1):
        xnew = []
        Tnew = []
        nnew = 1
        xnew.append(xmin)
        Tnew.append(T[0])
        while (xnew[nnew-1] < xmax - hmin):
            for i in range(0, NX-1):
                if (xnew[nnew-1] >= x[i] and xnew[nnew-1] <= x[i+1] and xnew[nnew-1] < xmax - hmin):
                    hll = (hloc[i] * (x[i+1] - xnew[nnew-1]) + hloc[i+1] * (xnew[nnew-1] - x[i])) / (x[i+1] - x[i])
                    hll = min(max(hmin, hll), hmax)
                    nnew += 1
                    xnew.append(min(xmax, xnew[nnew-2] + hll))
                    # Interpolation de la solution pour l'initialisation
                    un = (T[i] * (x[i+1] - xnew[nnew-1]) + T[i+1] * (xnew[nnew-1] - x[i])) / (x[i+1] - x[i])
                    Tnew.append(un)

        NX0 = NX
        NX = nnew
        x = np.linspace(xmin, xmax, NX)
        x[0:NX] = xnew[0:NX]
        T = np.zeros((NX))

    # Initialisation des tableaux
    rest = []
    F = np.zeros((NX))
    RHS = np.zeros((NX))
    hloc = np.ones((NX)) * hmax * 0.5
    metric = np.zeros((NX))
    Tex = np.zeros((NX))

    # Calcul de la solution exacte (ou de référence) Tex
    for j in range(1, NX-1):
        Tex[j] = np.exp(-20 * (x[j] - (xmax + xmin) * 0.5)**2)

    # Calcul du terme source F (dépendant du temps)
    dt = 1.e30
    for j in range(1, NX-1):
        Tx = (Tex[j+1] - Tex[j-1]) / (x[j+1] - x[j-1])
        Txip1 = (Tex[j+1] - Tex[j]) / (x[j+1] - x[j])
        Txim1 = (Tex[j] - Tex[j-1]) / (x[j] - x[j-1])
        Txx = (Txip1 - Txim1) / (0.5 * (x[j+1] + x[j]) - 0.5 * (x[j] + x[j-1]))
        F[j] = V * Tx - K * Txx + lamda * Tex[j]
        dt = min(dt, 0.25 * (x[j+1] - x[j-1])**2 / (V * np.abs(x[j+1] - x[j-1]) + 4 * K + np.abs(F[j]) * (x[j+1] - x[j-1])**2))

    print('NX =', NX, 'Dt =', dt)

    # Boucle temporelle
    n = 0
    res = 1
    res0 = 1
    t = 0
    solutions = []  # Pour stocker les solutions à différents instants
    times = [0.25, 0.5, 0.75, 1.0]  # Instants cibles pour la visualisation

    while (n < NT and t < Time):
        n += 1
        dt = min(dt, Time - t)
        t += dt

        # Discrétisation de l'équation ADRS
        res = 0
        for j in range(1, NX-1):
            visnum = 0.25 * (0.5 * (x[j+1] + x[j]) - 0.5 * (x[j] + x[j-1])) * np.abs(V)
            xnu = K + visnum
            Tx = (T[j+1] - T[j-1]) / (x[j+1] - x[j-1])
            Txip1 = (T[j+1] - T[j]) / (x[j+1] - x[j])
            Txim1 = (T[j] - T[j-1]) / (x[j] - x[j-1])
            Txx = (Txip1 - Txim1) / (0.5 * (x[j+1] + x[j]) - 0.5 * (x[j] + x[j-1]))

            # Terme source dépendant du temps : u(t) = sin(4*pi*t)
            u_t = 4 * np.pi * np.cos(4 * np.pi * t)
            src = u_t * Tex[j] + Tex[j] * np.cos(4 * np.pi * t) * 4 * np.pi

            RHS[j] = dt * (-V * Tx + xnu * Txx - lamda * T[j] + src)
            metric[j] += min(1. / hmin**2, max(1. / hmax**2, abs(Txx) / err))
            res += abs(RHS[j])

        metric[0] = metric[1]
        metric[NX-1] = metric[NX-2]

        for j in range(1, NX-1):
            T[j] += RHS[j]
            RHS[j] = 0

        # Conditions aux limites
        T[0] = 0
        T[NX-1] = 2 * T[NX-2] - T[NX-3]

        # Stockage des solutions à des instants cibles
        for target_time in times:
            if abs(t - target_time) < dt/2:
                solutions.append((t, T.copy()))

        if (n == 1):
            res0 = res
        rest.append(res)

    # Calcul de l'erreur L2
    errL2 = 0.0
    for j in range(1, NX-1):
        errL2 += (0.5 * (x[j+1] - x[j-1])) * (T[j] - Tex[j])**2
    errL2 = np.sqrt(errL2)
    errorL2[iter-1] = errL2

    # Critère d'arrêt mixte
    if (NX >= NX_min) and (errorL2[iter-1] < epsilon_L2):
        print(f"Critère d'arrêt atteint : NX = {NX} >= {NX_min} et erreur L2 = {errorL2[iter-1]:.4f} < {epsilon_L2}")
        break

    # Moyenne de la métrique sur le temps
    metric[0:NX] /= n
    hloc[0:NX] = np.sqrt(1. / metric[0:NX])

    print('Iteration =', iter, 'NX =', NX, 'Erreur L2 =', errorL2[iter-1], 'Temps =', t)

# Visualisation des solutions à différents instants
plt.figure()
for t_sol, T_sol in solutions:
    plt.plot(x, T_sol, label=f"t = {t_sol:.2f}")
plt.xlabel(u'$x$', fontsize=14)
plt.ylabel(u'$T$', fontsize=14, rotation=0)
plt.title(u'Solution à différents instants')
plt.legend()
plt.show()

# Visualisation de la convergence de l'erreur L2
plt.figure()
plt.plot(itertab[0:iter], errorL2[0:iter], label="Erreur L2")
plt.axhline(y=epsilon_L2, color='r', linestyle='--', label=f"Seuil L2 = {epsilon_L2}")
plt.xlabel("Itération d'adaptation")
plt.ylabel("Erreur L2")
plt.legend()
plt.title("Convergence de l'erreur L2")
plt.show()
