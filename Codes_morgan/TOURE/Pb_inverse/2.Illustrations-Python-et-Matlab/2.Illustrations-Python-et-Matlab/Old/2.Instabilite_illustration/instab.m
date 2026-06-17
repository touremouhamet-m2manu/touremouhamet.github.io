clear all ; close all ; 

fprintf('\n\n\n') ;  
fprintf('\n *********** Illustration de l''instabilite ************\n') ;

% Transformation

n = 256 ; 

for i = 1:n
    for j = 1:n
    A(i,j)=  exp(-0.001*(i-j)^2) ; 
    end
end
A = A + eye(n,n) ; 
condA  = cond(A) 
figure(1) ;
colormap(hot(256)) ; 
imshow(A) ;
title('matrice de la transformation') ;
colormap(hot(256)) ; 
pause


% fonction d'entree

h = 1/n ; 
x = h:h:1 ;
f0 = 2*x.^2.*(1-x.^2) -  x.^4.*(1-x.^4) - x.^6.*(1-x.^6)  ; 
figure(2) ;
subplot(121) ;
plot(x,f0) ;
title('fonction d''entree') ;
pause


% Fonction de sortie

b0 = A*f0' ;

figure(2) ;
subplot(122) ;
plot(x,b0) ;
title('fonction de sortie') ;
pause


% Inversion 

f_rec = A\b0 ; 
figure(3) ;
plot(x,f0,x,f_rec) ;
title('fonction d''entree reconstruite') ;
pause

% perturbations de l'entree

Ne = 70 ;
Ampl = .5 ; % amplitude
for freq = 1:Ne
    
    f = f0 + Ampl*sin(2*pi*freq*x) ; % ajout perturbation 
    
    figure(4) ;
    subplot(221) ;
    plot(x,f0) ; 
    title('entree d''origine') ; 
    
    b = A*f' ;
    
    subplot(222) ;
    plot(x,b0) ; 
    title('sortie d''origine') ; 
    
    subplot(223) ;
    plot(x,f) ; 
    title('entree perturbee') ; 
    
    subplot(224) ;
    plot(x,b) ; 
    title('sortie perturbee') ; 
    pause
    
end



% reconstruction avec perturbation

fprintf('reconstruction faussee\n') ;

f_rec = A\b ;

figure(5) ;
subplot(222) ;
plot(x,b0) ;
title('donnee fiable') ;
subplot(224) ;
plot(x,b) ;
title('donnee erronee') ;
subplot(221) ;
plot(x,f0) ; 
title('solution esperee') ; 
subplot(223) ;
plot(x,f_rec) ; 
title('solution trouvée') ;


erreur_donnee = norm(b-b0)/norm(b)*100 

erreur_sol = norm(f0-f_rec')/norm(f0)*100 

erreur_max = erreur_donnee*condA


