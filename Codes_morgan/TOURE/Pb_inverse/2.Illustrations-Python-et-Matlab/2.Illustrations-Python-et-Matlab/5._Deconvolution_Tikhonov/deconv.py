#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#from math import *
from numpy import *
#from scipy import * 
from matplotlib import * 
from matplotlib.pyplot import *
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%
print("",end='\n\n') 
print("** Déconvolution régularisée par Tikhonov-Phillips **", end='\n\n') 

n = 64
A = zeros((n,n))
for i in range(n): 
    for j in range(n):
        A[i,j] =  exp(-0.01*(i-j)**2)  
        #A[i,j] =  n/(i+j+2) 
A = A + 1e-10*eye(n) 
print("Conditionnement de A = {: .2e}".format(linalg.cond(A) ), end='\n\n') 
imshow(A) 
title('matrice de convolution')
show()

# fonction d'entrée
h = 1/n  
x = linspace(0,1,n)
f = 1*sin(2*(2*pi*x)) + 2*x**2*(1-x**2) - 8*x**4*(1-x**4) - 4*x**6*(1-x**6)  
subplot(121) 
plot(x,f) 
title('signal d entrée') 

# Fonction de sortie
g = dot(A,f)

subplot(122) 
plot(x,g) 
title('signal de sortie')
show()

# Inversion 
f_rec = dot(linalg.inv(A), g)   # SANS PASSER PAR FOURIER 
plot(x,f,x,f_rec)
title('signal obtenu par deconvolution directe matricielle, sans passer par Fourier...') 
show()
err_sol = linalg.norm(f-f_rec)/linalg.norm(f)*100 
print("Erreur sur la reconstruction = {: .2f} %".format(err_sol), end='\n\n') 

#%%% REGULARISATION DE TIKONOV-PHILLIPS %%%
epsilon = 1e-7
B_eps = linalg.inv(transpose(A)*A + epsilon*eye(n)) * transpose(A)  # pseudo inverse de Tikonov-Phillips
print("Conditionnement de B_eps = {: .2e}".format(linalg.cond(B_eps)), end='\n\n')
f_eps = dot(B_eps,g)*1/2   # SANS PASSER PAR FOURIER 
plot(x,f,x,f_eps) 
title('signal reconstruit par déconvolution régularisée TP matricielle, sans passer par Fourier...')
show()
err_sol = linalg.norm(f-f_eps)/linalg.norm(f)*100 
print("Erreur après régularisation = {: .2f} %".format(err_sol), end='\n\n') 