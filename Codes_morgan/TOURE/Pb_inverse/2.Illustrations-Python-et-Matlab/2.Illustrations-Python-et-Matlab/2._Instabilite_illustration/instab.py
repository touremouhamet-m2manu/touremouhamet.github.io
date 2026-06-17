#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from math import *
from numpy import *
#from scipy import *
from matplotlib import * 
from matplotlib.pyplot import *
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("",end='\n') 
print("Illustration de l'instabilité", end='\n')

n = 256  
A = zeros((n,n))

for i in range(0,n-1):
    for j in range(0,n-1):
        A[i,j]=  exp(-0.001*(i-j)**2) ; 
    
A = A + eye(n,n) ; 

condA  = linalg.cond(A) 
print("",end='\n') 
print("Conditionemment de la matrice = {: .2f}".format(condA), end='\n\n')

fig1 = imshow(A) 
show()
title ('Matrice de l application')
input("press any key to continue...")

# fonction d'entrée

x = linspace(0,1,n) 
f0 = 2**x**2*(1-x**2) -  x**4*(1-x**4) - 2*x**6*(1-x**6)  
b0 = dot(A,f0)
# Inversion 
f_rec = linalg.solve(A,b0)  

# Plots d'origine
fig, (ax1, ax2, ax3) = subplots(1, 3)
ax1.plot(x,f0) 
ax1.set_title('Signal f d entrée')
ax2.plot(x,b0,'r')
ax2.set_title('Signal Af de sortie')
ax3.plot(x,f_rec,'g') 
ax3.set_title('Signal d entrée reconstruit')
show()
#close(fig)

#  perturbations de l'entree
Nf = 20
ampl = 0.5  # amplitude
for freq in range(1,Nf):
    f = f0 + ampl*sin(2*pi*freq*2*x)  # ajout perturbation     
    fig, (ax1, ax2) = subplots(1, 2)
    #fig.suptitle('fonction d entrée perturbée et sortie correspondante')
    b = dot(A,f)
    erreur_f = linalg.norm(f0-f)/linalg.norm(f0)*100 
    print("Perturbation sur l entrée = {: .2f} %".format(erreur_f), end='\n')
    erreur_donnee = linalg.norm(b-b0)/linalg.norm(b)*100 
    print("Erreur sur la sortie (donnée) = {: .2f} %".format(erreur_donnee), end='\n\n')
    ax1.plot(x,f0,x,f)
    ax1.set_title('Entrée perturbée')
    ax2.plot(x,b0,x,b,'r') 
    ax2.set_title('sortie perturbee') 
    pause(4)
    close()
    show()
    #close(fig)

#  reconstruction avec perturbation
print("",end='\n') 
print('Reconstruction faussée\n') ;

f_rec = linalg.solve(A,b)

plot(x,f0,x,f_rec,'g')
title('entrée d orgine et solution trouvée')
show()

erreur_donnee = linalg.norm(b-b0)/linalg.norm(b)*100 
print("Erreur sur la donnée = {: .2f} %".format(erreur_donnee), end='\n\n')
erreur_sol = linalg.norm(f0-f_rec)/linalg.norm(f0)*100 
print("Erreur sur la solution = {: .2f} %".format(erreur_sol), end='\n\n')
erreur_max = erreur_donnee*condA
print("Erreur maximale possible = {: .2f} %".format(erreur_max), end='\n\n')

