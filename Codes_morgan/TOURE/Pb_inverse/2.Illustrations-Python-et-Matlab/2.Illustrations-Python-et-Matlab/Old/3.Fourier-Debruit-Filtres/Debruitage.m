
clear all ; close all ; 

% LE SIGNAL

L = 10 ; % longueur du signal
h = 1/100 ; % pas d'echantillonage 
x = (0:h:L); % subdivision de [0,L]
f0 = sin(2*x).*sin(1.0*x.^2) ;


% AJOUT DE BRUIT 

sigma = 0.10 ; 
f = f0 + sigma*randn(1,L/h+1) ; % bruit gaussien d'ecart-type sigma
figure(1); 
pause; subplot(2,3,1); plot(x,f0); title('SIGNAL D''ORIGINE') ;
RBSf = norm(f-f0)/norm(f0)*100 ;
fprintf('\n');
fprintf('Rapport bruit / signal = %2.1f %%\n',RBSf);
fprintf('\n');
pause; subplot(2,3,2); plot(x,f); title('SIGNAL BRUITE') ;


% TF de f 

F = fft(f);
pause; subplot(2,3,3); plot(x(1:500),abs(F(1:500))); title('TRANSFORMEE DE FOURIER') ;


% FILTRAGE DES HAUTES FREQUENCES (BRUIT...)

H = rectpuls(x)+rectpuls((L-x));
Fc = F.*H; 
pause; subplot(2,3,6); plot(x(1:500),abs(Fc(1:500)),x(1:500),90*H(1:500)); title('TF TRONQUEE ET FILTRE') ;


% TF INVERSE 

fd = ifft(Fc);


% ECRITURE DES RESULTATS

pause; subplot(2,3,5); plot(x,f,x,real(fd)); title('SIGNAL BRUITE ET DEBRUITE') ;
pause; subplot(2,3,4); plot(x,f0,x,real(fd)); title('SIGNAL D''ORIGINE ET DEBRUITE') ;
RBSfd = norm(fd-f)/norm(f)*100 ;
RBSfd0 = (1-norm(fd-f0)/norm(f0))*100 ;
fprintf('Taux de debruitage = %2.1f %%\n',RBSfd);
fprintf('\n');
fprintf('Fiabilite de la reconstruction = %2.1f %%\n',RBSfd0);
fprintf('\n');
pause ;
figure(2); 
pause; plot(x,f0,x,real(fd)); title('SIGNAL D''ORIGINE ET DEBRUITE') ;
pause ;
figure(3); 
pause; plot(x,f,x,real(fd)); title('SIGNAL BRUITE ET DEBRUITE') ;
