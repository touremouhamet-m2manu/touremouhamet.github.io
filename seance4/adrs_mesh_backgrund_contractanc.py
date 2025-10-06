import numpy as np
import matplotlib.pyplot as plt

# Paramètres physiques
K = 0.01     # Coefficient de diffusion
xmin = 0.0
xmax = 1.0
Time = 10.0  # Temps d'intégration
V = 1.0
lamda = 1.0

# Paramètres de maillage
niter_refinement = 30  # Nombre d'itérations de raffinement
hmin = 0.02
hmax = 0.15
err = 0.03

# Paramètres numériques
NT = 10000   # Nombre maximal d'itérations temporelles
ifre = 1000000  # Fréquence de traçage
eps = 0.001     # Ratio de convergence relative

# Fonction pour la solution exacte
def Texact(x):
    return 2*np.exp(-100*(x-(xmax+xmin)*0.25)**2) + np.exp(-200*(x-(xmax+xmin)*0.65)**2)

# Initialisation
NX = 3    # Nombre initial de points de maillage
errorL2 = np.zeros((niter_refinement))
errorH1 = np.zeros((niter_refinement))
itertab = np.zeros((niter_refinement))
hloc = np.ones((NX))*hmax
itera = 0
NX0 = 0

# Maillage de fond (background mesh)
background_mesh = np.linspace(xmin, xmax, 10000)
T_history = []

while np.abs(NX0-NX) > 2 and itera < niter_refinement-1:
    itera += 1
    itertab[itera] = 1./NX

    # Initialisation du maillage
    x = np.linspace(xmin, xmax, NX)
    T = np.zeros((NX))

    # Adaptation du maillage en utilisant la métrique locale
    if itera > 0:
        xnew = []
        Tnew = []
        nnew = 1
        xnew.append(xmin)
        Tnew.append(T[0])

        while xnew[nnew-1] < xmax-hmin:
            for i in range(0, NX-1):
                if xnew[nnew-1] >= x[i] and xnew[nnew-1] <= x[i+1] and xnew[nnew-1] < xmax-hmin:
                    hll = (hloc[i]*(x[i+1]-xnew[nnew-1])+hloc[i+1]*(xnew[nnew-1]-x[i]))/(x[i+1]-x[i])
                    hll = min(max(hmin, hll), hmax)
                    nnew += 1
                    xnew.append(min(xmax, xnew[nnew-2]+hll))
                    un = (T[i]*(x[i+1]-xnew[nnew-1])+T[i+1]*(xnew[nnew-1]-x[i]))/(x[i+1]-x[i])
                    Tnew.append(un)

        NX0 = NX
        NX = nnew
        x = np.zeros((NX))
        x[0:NX] = xnew[0:NX]
        T = np.zeros((NX))
        T[0:NX] = Tnew[0:NX]

    # Calcul de la solution exacte sur le maillage actuel
    Tex = np.zeros((NX))
    for j in range(1, NX-1):
        Tex[j] = Texact(x[j])

    # Calcul du pas de temps
    dt = 1.e30
    F = np.zeros((NX))
    RHS = np.zeros((NX))
    metric = np.ones((NX))
    hloc = np.ones((NX))*hmax*0.5

    for j in range(1, NX-1):
        Tx = (Tex[j+1]-Tex[j-1])/(x[j+1]-x[j-1])
        Txip1 = (Tex[j+1]-Tex[j])/(x[j+1]-x[j])
        Txim1 = (Tex[j]-Tex[j-1])/(x[j]-x[j-1])
        Txx = (Txip1-Txim1)/(0.5*(x[j+1]+x[j])-0.5*(x[j]+x[j-1]))
        F[j] = V*Tx - K*Txx + lamda*Tex[j]
        dt = min(dt, 0.5*(x[j+1]-x[j-1])**2/(V*np.abs(x[j+1]-x[j-1])+4*K+np.abs(F[j])*(x[j+1]-x[j-1])**2))

    print('NX=', NX, 'Dt=', dt)

    # Intégration temporelle
    n = 0
    res = 1
    res0 = 1
    t = 0
    rest = []

    while n < NT and res/res0 > eps and t < Time:
        n += 1
        t += dt

        # Discrétisation de l'équation d'advection/diffusion/réaction/source
        res = 0
        for j in range(1, NX-1):
            visnum = 0.5*(0.5*(x[j+1]+x[j])-0.5*(x[j]+x[j-1]))*np.abs(V)
            xnu = K + visnum
            Tx = (T[j+1]-T[j-1])/(x[j+1]-x[j-1])
            Txip1 = (T[j+1]-T[j])/(x[j+1]-x[j])
            Txim1 = (T[j]-T[j-1])/(x[j]-x[j-1])
            Txx = (Txip1-Txim1)/(0.5*(x[j+1]+x[j])-0.5*(x[j]+x[j-1]))
            RHS[j] = dt*(-V*Tx + xnu*Txx - lamda*T[j] + F[j])
            metric[j] = min(1./hmin**2, max(1./hmax**2, abs(Txx)/err))
            res += abs(RHS[j])

        metric[0] = metric[1]
        metric[NX-1] = metric[NX-2]

        for j in range(0, NX-1):
            metric[j] = 0.5*(metric[j]+metric[j+1])
        metric[NX-1] = metric[NX-2]

        hloc[0:NX] = np.sqrt(1./metric[0:NX])

        for j in range(1, NX-1):
            T[j] += RHS[j]
            RHS[j] = 0

        T[NX-1] = T[NX-2]

        if n == 1:
            res0 = res
        rest.append(res)

        if n % ifre == 0 or (res/res0) < eps:
            print('iter=', n, 'residual=', res)

    print('iter=', n, 'time=', t, 'residual=', res)

    # Calcul des erreurs L2 et H1
    errL2h = 0
    errH1h = 0
    for j in range(1, NX-1):
        Texx = (Tex[j+1]-Tex[j-1])/(x[j+1]-x[j-1])
        Tx = (T[j+1]-T[j-1])/(x[j+1]-x[j-1])
        errL2h += (0.5*(x[j+1]+x[j])-0.5*(x[j]+x[j-1]))*(T[j]-Tex[j])**2
        errH1h += (0.5*(x[j+1]+x[j])-0.5*(x[j]+x[j-1]))*(Tx-Texx)**2

    errorL2[itera] = errL2h
    errorH1[itera] = errL2h + errH1h

    print('norm error L2, H1=', errL2h, errH1h)
    print('----------------------------------')

    # Interpolation sur le maillage de fond
    T_background = np.interp(background_mesh, x, T)
    T_history.append(T_background)

    # Calcul de la contraction
    if itera > 1:
        contraction = np.max(np.abs(T_history[-1] - T_history[-2]))
        print(f"Iteration {itera}: Contraction={contraction:.6f}")

# Tracer les résultats
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(background_mesh, Texact(background_mesh), label='Solution exacte')
plt.plot(background_mesh, T_history[-1], '--', label='Solution adaptée')
plt.xlabel('x')
plt.ylabel('T(x)')
plt.title('Solution adaptée vs exacte')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
for i, T_bg in enumerate(T_history):
    plt.plot(background_mesh, T_bg, label=f'Itération {i+1}')
plt.xlabel('x')
plt.ylabel('T(x)')
plt.title('Évolution de la solution sur le maillage de fond')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('solution_evolution.png')
plt.show()

# Tracer la contraction
contractions = []
for i in range(1, len(T_history)):
    contractions.append(np.max(np.abs(T_history[i] - T_history[i-1])))

plt.figure(figsize=(12, 6))
plt.semilogy(contractions, 'bo-', label='Contraction')
plt.xlabel('Itération')
plt.ylabel('Contraction (log)')
plt.title('Évolution de la contraction')
plt.grid(True)
plt.legend()
plt.savefig('contraction_evolution.png')
plt.show()

# Tracer l'erreur en fonction du nombre de points
plt.figure(figsize=(12, 6))
plt.loglog(itertab[1:itera], errorL2[1:itera], 'bo-', label='Erreur L2')
plt.xlabel('1/NX')
plt.ylabel('Erreur L2 (log)')
plt.title('Erreur L2 en fonction du nombre de points')
plt.grid(True)
plt.legend()
plt.savefig('error_vs_NX.png')
plt.show()
