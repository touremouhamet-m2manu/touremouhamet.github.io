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
print("Filtres PB et PH de la somme de 2 ondes pures", end='\n\n')

# somme de 2 ondes pures
N = 1024
h = 1/N
t = linspace(0,1,N)
signal = 3*cos(2*(2*pi*t))
bruit = cos(50*(2*pi*t))
f = signal + bruit

# FILTRE PASSE BAS
subplot(2,2,1) 
plot(t,f)
title('Signal original "bruité"')
F = abs(fft(f))
subplot(2,2,2) 
plot(t[1:200],F[1:200]/N,'r')
title('TF du signal (amplitude)')
H = piecewise(t, [t <= 0.03, t >= 0.03 ], [1, 0])
F_pb = F * H 
subplot(2,2,4) 
plot(t[1:200],F_pb[1:200]/N,'r',t[1:200],H[1:200],'g')
title('TF coupée par le filtre passe bas')
f_pb = real(ifft(F_pb)) * 2  # perte de la partie symétrique de la TF par le module...
subplot(2,2,3) 
plot(t,f_pb,t,signal,'m')
title('Signal filtré bas et original')
show()

# FILTRE PASSE HAUT (OU BANDE...)
subplot(2,2,1) 
plot(t,f)
title('Signal original "bruité"')
F = abs(fft(f))
subplot(2,2,2) 
plot(t[1:200],F[1:200]/N,'r')
title('TF')
#H = piecewise(t, [t >= 0.04, t <= 0.04], [1, 0]) # Filtre passe haut 
H = piecewise(t, [t >= 0.045, t <= 0.045], [1, 0]) - piecewise(t, [t >= 0.055, t <= 0.055], [1, 0]) # Passe bande...
F_ph = F * H 
print(F_ph)
subplot(2,2,4) 
plot(t[1:200],F_ph[1:200]/N,'r',t[1:200],H[1:200],'g')
title('TF coupée par le filtre passe haut (ou bande)')
f_ph = real(ifft(F_ph)) * 2  # perte de la partie symétrique de la TF par le module...
subplot(2,2,3) 
plot(t,f_ph,t,bruit,'k')
title('Signal filtré haut et "bruit"')
show()
close('all')