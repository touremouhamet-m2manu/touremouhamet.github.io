clear all ; close all ; 


% somme de 2 ondes pures
t = (0:1/512:1);
f = 2*cos(2*(2*pi*t))+ cos(30*(2*pi*t));
fo = 2*cos(2*(2*pi*t));
br = cos(30*(2*pi*t));

% PASSE BAS
figure; subplot(2,2,1); plot(t(1:512),f(1:512));
pause;
F = abs(fft(f));
subplot(2,2,2); plot(F(1:40));
H = 1.0*rectpuls(15*t);
Fpb = F.*H; 
pause; subplot(2,2,4); plot(Fpb(1:40));
fpb = real(ifft(Fpb));
pause; subplot(2,2,3); plot(t(1:512),fpb(1:512));

% PASSE HAUT
pause;
figure; subplot(2,2,1); plot(t(1:512),f(1:512));
pause;
F = abs(fft(f));
subplot(2,2,2); plot(F(1:40));
H = 1.0*rectpuls(7*t)-1.0*rectpuls(15*t);
Fph = F.*H; 
pause; subplot(2,2,4); plot(Fph(1:40));
fph = real(fft(Fph));
pause; subplot(2,2,3); plot(t(1:512),fph(1:512));


pause


%%%%% ESSAI AVEC UN SIGNAL SONORE %%%%%

% [y,Fs] = audioread('HornsNew.wav');

load chirp.mat
filename = 'chirp.wav';
audiowrite(filename,y,Fs);
[y,Fs] = audioread('chirp.wav');
[y,Fs] = audioread('handel.wav');
f = y' ;
sound(f,Fs);

L = length(f) ;
t = (1/L:1/L:1);


% PASSE BAS
figure; subplot(2,2,1); plot(t,f);
pause;
F = abs(fft(f));
subplot(2,2,2); plot(F);
H = 1.0*rectpuls(1.5*t);
Fpb = F.*H; 
pause; subplot(2,2,4); plot(Fpb);
fpb = real(ifft(Fpb));
pause; subplot(2,2,3); plot(t,fpb);

pause
sound(fpb,Fs);
pause

% PASSE HAUT
pause;
figure; subplot(2,2,1); plot(t,f);
pause;
F = abs(fft(f));
subplot(2,2,2); plot(F);
H = 1.0 - 1.0*rectpuls(1.5*t);
Fph = F.*H; 
pause; subplot(2,2,4); plot(Fph);
fph = real(fft(Fph));
pause; subplot(2,2,3); plot(t,fph);

pause

sound(fph,Fs);
pause

