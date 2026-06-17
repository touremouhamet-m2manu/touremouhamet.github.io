%  convolution répetée de la porte 

clear all ; close all ;  

N = 200 ; 
L = 5 ; 
h = 2*L/N ; 


t = (-L:h:L); 
f = rectpuls(t);
s = f;
figure;

for i = 1:16
subplot(4,4,i); plot(t,s);
pause;
s = conv(s,f,'same') * h ; % integrale de Riemann -> pas h

integ = sum(s)*h % valeur de l'integrale

end;

