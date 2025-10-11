import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1️⃣  Fonction exacte instationnaire et source forcée
# ------------------------------------------------------
def fex(NX, dx, time, V, K, lamda):
    F = np.zeros((NX))
    Tex = np.zeros((NX))
    Text = np.zeros((NX))
    Texx = np.zeros((NX))
    
    for j in range(1, NX-1):
        # fonction spatiale de référence avec double gaussienne
        v = (np.exp(-1000*((j - NX/3)/NX)**2) + np.exp(-10*np.exp(-1000*((j - NX/3)/NX)**2))) * np.sin(5*j*math.pi/NX)
        Tex[j] = np.sin(4*math.pi*time) * v
        Text[j] = 4*math.pi*np.cos(4*math.pi*time) * v  # dérivée temporelle exacte

    for j in range(1, NX-1):
        # dérivées spatiales discrètes
        Texx[j] = (Tex[j+1]-Tex[j-1])/(2*dx)
        Txx = (Tex[j+1]-2*Tex[j]+Tex[j-1])/(dx**2)
        F[j] = V*Texx[j] - K*Txx + lamda*Tex[j] + Text[j]  # source forcée
    return F, Tex, Texx

# ------------------------------------------------------
# 2️⃣  Paramètres physiques
# ------------------------------------------------------
K = 0.1     # Coefficient de diffusion
L = 1.0     # Taille du domaine
Time = 1.0  # Temps final
V = 1.0
lamda = 1.0

# ------------------------------------------------------
# 3️⃣  Paramètres numériques globaux
# ------------------------------------------------------
NT = 10000
ifre = 100
eps = 0.001
niter_refinement = 10

irk_max = 4   # On testera Runge-Kutta 1 à 4
alpha = np.zeros(irk_max)
for irk in range(irk_max):
    alpha[irk] = 1.0 / (irk_max - irk)

# ------------------------------------------------------
# 4️⃣  Étude 1 : Erreur à T/2 et Tfin pour différents maillages
# ------------------------------------------------------
NX_tab = []
Err_tab1 = []
Err_tab2 = []

for iter in range(niter_refinement):
    NX = 5 + 3*iter  # on augmente progressivement la finesse du maillage
    dx = L / (NX - 1)
    dt = dx**2 / (V*dx + K + dx**2)

    x = np.linspace(0.0, 1.0, NX)
    T = np.zeros(NX)

    n = 0
    time = 0.0
    time_total = Time
    error_iter = 0.0

    while time < time_total:
        n += 1
        F, Tex, Texx = fex(NX, dx, time, V, K, lamda)
        dt = dx**2 / (V*dx + 2*K + abs(np.max(F))*dx**2)
        time += dt

        T0 = T.copy()
        # On applique un schéma de Runge-Kutta 2 (ordre 2) ici
        for irk in range(irk_max):
            for j in range(1, NX-1):
                xnu = K + 0.5*dx*abs(V)
                Tx = (T[j+1]-T[j-1])/(2*dx)
                Txx = (T[j-1]-2*T[j]+T[j+1])/(dx**2)
                RHS = dt * (-V*Tx + xnu*Txx - lamda*T[j] + F[j])
                T[j] = T0[j] + RHS * alpha[irk]

        # calcul erreur L2 à l’instant courant
        err_L2 = np.sqrt(np.dot(T - Tex, T - Tex) * dx)

        if abs(time - 0.5*Time) < dt/2:
            Err_tab1.append(err_L2)
        if abs(time - Time) < dt/2:
            Err_tab2.append(err_L2)

    NX_tab.append(NX)
    print(f"NX={NX:3d}  |  Err(T/2)={Err_tab1[-1]:.3e}  |  Err(Tfin)={Err_tab2[-1]:.3e}")

# ✅ Visualisation de l'erreur à T/2 et Tfin
plt.figure(figsize=(7,5))
plt.plot(NX_tab, Err_tab1, 'o--', label='Erreur à T/2')
plt.plot(NX_tab, Err_tab2, 's--', label='Erreur à Tfin')
plt.xlabel("Nombre de points NX")
plt.ylabel("Erreur L²")
plt.title("Erreur L² à T/2 et Tfin pour différents maillages")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------
# 5️⃣  Étude 2 : Évolution de l’erreur au point milieu pour RK1–4
# ------------------------------------------------------
plt.figure(figsize=(8,6))
colors = ['r', 'g', 'b', 'k']
labels = ['RK1 (Euler)', 'RK2', 'RK3', 'RK4']

NX = 101  # maillage fixe pour la comparaison
dx = L / (NX - 1)
x = np.linspace(0.0, 1.0, NX)
mid_idx = NX // 2  # point milieu

for irk_order in range(1, irk_max+1):
    alpha_rk = np.zeros(irk_order)
    for irk in range(irk_order):
        alpha_rk[irk] = 1.0 / (irk_order - irk)

    T = np.zeros(NX)
    time = 0.0
    dt = dx**2 / (V*dx + K + dx**2)
    time_tab = []
    err_mid = []

    while time < Time:
        F, Tex, Texx = fex(NX, dx, time, V, K, lamda)
        dt = dx**2 / (V*dx + 2*K + abs(np.max(F))*dx**2)
        T0 = T.copy()
        for irk in range(irk_order):
            for j in range(1, NX-1):
                xnu = K + 0.5*dx*abs(V)
                Tx = (T[j+1]-T[j-1])/(2*dx)
                Txx = (T[j-1]-2*T[j]+T[j+1])/(dx**2)
                RHS = dt * (-V*Tx + xnu*Txx - lamda*T[j] + F[j])
                T[j] = T0[j] + RHS * alpha_rk[irk]

        time += dt
        time_tab.append(time)
        # erreur au point milieu
        err_mid.append(abs(T[mid_idx] - Tex[mid_idx]))

    plt.plot(time_tab, err_mid, color=colors[irk_order-1], label=labels[irk_order-1])

plt.xlabel("Temps")
plt.ylabel("Erreur |T(x=L/2,t) - Tex|")
plt.title("Évolution temporelle de l’erreur au point milieu pour différents RK")
plt.legend()
plt.grid(True)
plt.show()
