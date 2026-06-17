#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#from math import *
from numpy import *
#from scipy import * 
from matplotlib import * 
from matplotlib.pyplot import *
from numpy.fft import fft, ifft
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("",end='\n\n') 
print("Quelques transformées de Fourier", end='\n\n')

#################################################################
print("----------- Onde pure amortie --------------", end='\n\n') 
t = linspace(0,1,512) 
f = exp(-t**4)*cos(10*(2*pi*t))
F = abs(fft(f))
fig, (ax1, ax2) = subplots(2, 1)
ax1.plot(t,f) 
ax1.set_title('Signal f')
ax2.plot(t[1:256],F[1:256],'r') 
ylabel('Amplitude')
ax2.set_title('Transformée de Fourier F')
show()
#input("----")


#################################################################
print("----------- Onde pure amortie plus rapide --------------", end='\n\n') 
t = linspace(0,1,512) 
f = exp(-t**4)*cos(20*(2*pi*t))
F = abs(fft(f))
fig, (ax1, ax2) = subplots(2, 1)
ax1.plot(t,f) 
ax1.set_title('Signal f')
ax2.plot(t[1:256],F[1:256],'r') 
ylabel('Amplitude')
ax2.set_title('Transformée de Fourier F')
show()
#input("----")


#################################################################
print("----------- Somme de deux ondes pures --------------", end='\n\n') 
t = linspace(0,1,512) 
f = 2*cos(2*(2*pi*t)) + cos(20*(2*pi*t)) 
F = abs(fft(f))
fig, (ax1, ax2) = subplots(2, 1)
ax1.plot(t,f) 
ax1.set_title('Signal f')
ax2.plot(t[1:256],F[1:256],'r') 
ylabel('Amplitude')
ax2.set_title('Transformée de Fourier F')
show()
#input("----")


#################################################################
print("----------- Onde pure avec bruit additif --------------", end='\n\n') 
t = linspace(0,1,512) 
f = cos(3*(2*pi*t)) + random.normal(0, 1/2, 512)
F = abs(fft(f))
fig, (ax1, ax2) = subplots(2, 1)
ax1.plot(t,f) 
ax1.set_title('Signal f')
ax2.plot(t[1:256],F[1:256],'r') 
ylabel('Amplitude')
ax2.set_title('Transformée de Fourier F')
show()
#input("----")


#################################################################
print("----------- Produit de deux sinusoïdes--------------", end='\n\n') 
t = linspace(0,1,512) 
f =  cos(1*(2*pi*t)) * cos(30*(2*pi*t))
F = abs(fft(f))
fig, (ax1, ax2) = subplots(2, 1)
ax1.plot(t,f) 
ax1.set_title('Signal f')
ax2.plot(t[1:256],F[1:256],'r') 
ylabel('Amplitude')
ax2.set_title('Transformée de Fourier F')
show()
#input("----")


#################################################################
print("----------- Fonction porte --------------", end='\n\n') 
t = linspace(0,1,512) 
f = piecewise(t, [t >= 1/3, t >= 2/3 ], [1, 0])
F = abs(fft(f))
fig, (ax1, ax2) = subplots(2, 1)
ax1.plot(t,f) 
ax1.set_title('Signal f')
ax2.plot(t[1:256],F[1:256],'r') 
ylabel('Amplitude')
ax2.set_title('Transformée de Fourier F')
show()
#input("----")


#################################################################
print("----------- Bruit artificiel (Chirp) --------------", end='\n\n') 
t = linspace(0,1,512) 
f = cos(500*t**2+200*t**3)
F = abs(fft(f))
fig, (ax1, ax2) = subplots(2, 1)
ax1.plot(t,f) 
ax1.set_title('Signal f')
ax2.plot(t[1:256],F[1:256],'r') 
ylabel('Amplitude')
ax2.set_title('Transformée de Fourier F')
show()
#input("----")


#################################################################
print("----------- Gaussienne --------------", end='\n\n') 
t = linspace(0,1,512) 
f = exp(-5000*(t-.5)**2)
F = abs(fft(f))
fig, (ax1, ax2) = subplots(2, 1)
ax1.plot(t,f) 
ax1.set_title('Signal f')
ax2.plot(t[1:256],F[1:256],'r') 
ylabel('Amplitude')
ax2.set_title('Transformée de Fourier F')
show()
#input("----")

#################################################################
print("----------- Gaussienne plus contractée --------------", end='\n\n') 
t = linspace(0,1,512) 
f = exp(-10000*(t-.5)**2)
F = abs(fft(f))
fig, (ax1, ax2) = subplots(2, 1)
ax1.plot(t,f) 
ax1.set_title('Signal f')
ax2.plot(t[1:256],F[1:256],'r') 
ylabel('Amplitude')
ax2.set_title('Transformée de Fourier F')
show()
#input("----")