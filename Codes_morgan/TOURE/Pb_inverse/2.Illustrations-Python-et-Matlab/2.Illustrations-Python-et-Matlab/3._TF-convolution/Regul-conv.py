#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#from math import *
from numpy import *
#from scipy import * 
from matplotlib import * 
from matplotlib.pyplot import *
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("",end='\n\n') 
print("Effet régularisant de la Convolution", end='\n\n')

N = 1000 # Construction du signal
L = 4 
h = 2*L/N  
t = linspace(-L,L,N)
f0 = sin(2*t)*sin(0.3*t**2) 
f1 = 4*piecewise(t, [t > -1, t > 1 ], [1, 0])
f2 = 3*piecewise(t, [t > 1.5, t > 3.5 ], [1, 0]) * abs(2.5-t)
sigma = 0.1
bruit = sigma*random.normal(0, 1, N) # bruit gaussien
f = f0 + f1 + f2 + 0*bruit # signal irrégulier

for p in range(1,30):
    h0 = exp(-20*p*t**2) # filtre régularisant 
    h = h0 / linalg.norm(h0,1) #de norme 1 = 1 
    subplot(211)
    plot(t,f,t,2*h0)
    title('Signal irrégulier et convoluteur gaussien')
    s = convolve(f,h,'same') # convolution
    err_reg = linalg.norm(f-s)/linalg.norm(f)*100
    print("Erreur relative de la régularisation = {: .2f} %".format(err_reg), end='\n')
    subplot(212)
    plot(t,f,t,s,'r')
    title('Signal irrégulier et signal régularisé')
    pause(3)
    close()
    show()
