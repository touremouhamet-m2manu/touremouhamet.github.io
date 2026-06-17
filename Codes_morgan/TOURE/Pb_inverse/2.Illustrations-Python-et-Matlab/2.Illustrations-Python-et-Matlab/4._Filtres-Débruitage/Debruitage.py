#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#from math import *
from numpy import *
#from scipy import * 
from matplotlib import * 
from matplotlib.pyplot import *
from numpy.fft import fft, ifft
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%
print("",end='\n\n') 
print("Débruitage d'un signal par filtre passe bas", end='\n\n')

N = 1024 # nombre de points
L = 10  # longueur du signal
h = L/N  # pas d'echantillonage 
x = linspace(0,L,N) # subdivision de [0,L]
f0 = sin(2*x) * sin(0.03*x**2) # signal d'origine
sigma = 0.05
f = f0 + sigma*random.normal(0, 1, N) # ajout d'un bruit gaussien

subplot(2,3,1) 
plot(x,f0) 
title('SIGNAL D''ORIGINE') 

RBSf = linalg.norm(f-f0)/linalg.norm(f0)*100 
print("",end='\n') 
print("Rapport bruit / signal = {: .2f}".format(RBSf), end='\n\n')

subplot(2,3,2) 
plot(x,f) 
title('SIGNAL BRUITE') 

F = fft(f)

subplot(2,3,3) 
plot(x[1:500],abs(F[1:500]))
title('TRANSFORMEE DE FOURIER')

H = piecewise(x, [x <= .4, x >= .4], [1, 0]) # FILTRAGE DES HAUTES FREQUENCES / BRUIT...

Fc = F * H  
subplot(2,3,6) 
plot(x[1:500],abs(Fc[1:500]),x[1:500],90*H[1:500]) 
title('TF TRONQUEE ET FILTREE') 

fd = (ifft(Fc)) * 2 # TF INVERSE (multipliée par 2 car on a coupé les frequences négatives...) 

subplot(2,3,5) 
plot(x,f,x,fd)  
title('SIGNAL BRUITE ET DEBRUITE') 
subplot(2,3,4)
plot(x,f0,x,real(fd))
title('SIGNAL D''ORIGINE ET DEBRUITE') 
show()


RBSfd = linalg.norm(real(fd)-f0)/linalg.norm(f0)*100 
RBSfd0 = 100-RBSfd
print("Taux de debruitage = {: .2f} %".format(RBSfd), end='\n\n')
print("Fiabilite de la reconstruction = {: .2f} %".format(RBSfd0), end='\n\n')


plot(x,f0,x,real(fd)) 
title('SIGNAL D''ORIGINE ET DEBRUITE') 
show()
plot(x,f,x,real(fd)) 
title('SIGNAL BRUITE ET DEBRUITE')
show()