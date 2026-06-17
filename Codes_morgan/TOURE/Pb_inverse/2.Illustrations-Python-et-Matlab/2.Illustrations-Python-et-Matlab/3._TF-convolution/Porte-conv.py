#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#from math import *
from numpy import *
#from scipy import * 
from matplotlib import * 
from matplotlib.pyplot import *
from numpy.fft import fft, ifft
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("",end='\n\n') 
print("Convolutions successives d une porte", end='\n\n')

N = 200
L = 5 
h = 2*L/N  
t = linspace(-L,L,N)
f = piecewise(t, [t > -1, t > 1 ], [1, 0])
plot(t,f)
show()

s = f

for i in range(1,10):
    print(i)
    subplot(3,3,i)
    plot(t,s)
    pause(10-i)
    s = h * convolve(s,f,'same') # integrale de Riemann de pas h
    s = s /max(s)
    integ = h * sum(s)  # valeur de l'integrale
show()

