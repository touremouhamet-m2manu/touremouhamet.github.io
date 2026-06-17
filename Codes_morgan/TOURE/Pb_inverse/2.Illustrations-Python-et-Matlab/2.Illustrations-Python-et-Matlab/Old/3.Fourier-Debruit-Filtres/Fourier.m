
clear all ; close all ;

pause; % onde pure amortie
t = (0:1/512:1);
f = exp(-t.^2*4).* cos(20*(2*pi*t));
figure; ;subplot(1,2,1); plot(t(1:512),f(1:512)); title('SINUS AMORTI') ;
pause;
F = abs(fft(f,512));
subplot(1,2,2); plot(t(1:128),0,t(1:128),F(1:128)); title('TRANSFORMEE DE FOURIER') ;


pause; % onde pure amortie plus rapide
t = (0:1/512:1);

f = exp(-t.^2*4).* cos(60*(2*pi*t));
figure; subplot(1,2,1); plot(t(1:512),f(1:512)); title('...EN PLUS RAPIDE') ;
pause;
F = abs(fft(f));
subplot(1,2,2); plot(t(1:128),0,t(1:128),F(1:128)); title('TRANSFORMEE DE FOURIER') ;

pause; % somme de 2 ondes pures
t = (0:1/512:1);
f = 2*cos(2*(2*pi*t))+cos(20*(2*pi*t));
figure; subplot(1,2,1); plot(t(1:512),f(1:512)); title('SOMME DE DEUX SINUS') ;
pause;
F = abs(fft(f));
subplot(1,2,2); plot(t(1:32),0,t(1:32),F(1:32)); title('TRANSFORMEE DE FOURIER') ;

pause; % ondes pure + bruit
t = (0:1/512:1);
f = cos(3*(2*pi*t))+ randn(size(t));
figure; subplot(1,2,1); plot(t(1:512),f(1:512));title('SINUS AVEC BRUIT') ;
pause;
F = abs(fft(f));
subplot(1,2,2); plot(t(1:32),0,t(1:32),F(1:32)); title('TRANSFORMEE DE FOURIER') ;

pause; % produit de 2 ondes pures
t = (0:1/512:1);
f = cos(1*(2*pi*t)).*cos(30*(2*pi*t));
figure; subplot(1,2,1); plot(t(1:512),f(1:512)); title('PRODUIT DE DEUX SINUS') ;
pause;
F = abs(fft(f));
subplot(1,2,2); plot(t(1:64),0,t(1:64),F(1:64)); title('TRANSFORMEE DE FOURIER') ;


pause; % porte 
t = (0:1/512:1);
% f = (1-abs(t)).*square(12*t);
f = rectpuls(5*t);
figure; subplot(1,2,1); plot(t(1:512),f(1:512)); title('PORTE') ;
pause;
F = abs(fft(f));
subplot(1,2,2); plot(F(1:128)); title('TRANSFORMEE DE FOURIER') ;

pause; % chirp "chaotique"... bruit artificiel
t = (0:1/32:10);
f = cos(10*t.^2);
figure; subplot(1,2,1); plot(t(1:320),f(1:320)); title('GAZOUILLIS') ;
pause;
F = abs(fft(f));
subplot(1,2,2); plot(t(1:80),0,t(1:80),F(1:80)); title('TRANSFORMEE DE FOURIER') ;

pause;  % gaussienne
t = (0:1/512:1);
f = exp(-(t-.5).^2*20);  
figure; subplot(1,2,1); plot(t(1:512),f(1:512)); title('GAUSSIENNE') ;
pause;
F = abs(fft(f));
subplot(1,2,2); plot(t(1:32),0,t(1:32),F(1:32)); title('TRANSFORMEE DE FOURIER') ;

pause; % gaussienne plus contractée
t = (0:1/512:1);
f = exp(-(t-.5).^2*100);
figure; subplot(1,2,1); plot(t(1:512),f(1:512)); title('GAUSSIENNE') ;
pause;
F = abs(fft(f));
subplot(1,2,2); plot(t(1:32),0,t(1:32),F(1:32)); title('TRANSFORMEE DE FOURIER') ;


