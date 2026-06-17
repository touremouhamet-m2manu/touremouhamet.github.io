clear all ; close all ; 

fprintf('\n\n\n') ;  
fprintf('\n *********** Essai de deconvolution ************\n') ;

% convolution

n = 64 ; 

for i = 1:n
    for j = 1:n
    A(i,j)=  exp(-0.001*(i-j)^2) ; 
    end
end
 

A = A + 0.00000000001*eye(n,n) ; 

cond_A  = cond(A) 

figure(1) ;
colormap(hot) ; 
imshow(A) ;
title('matrice de convolution') ;
colormap(hot) ; 
pause


% fonction d'entree

h = 1/n ; 
x = h:h:1 ;
f = 1*sin(2*pi*2*x) + 2*x.^2.*(1-x.^2) -  8*x.^4.*(1-x.^4) - 4*x.^6.*(1-x.^6)  ; 
figure(2) ;
subplot(121) ;
plot(x,f) ;
title('signal d''entree') ;
pause


% Fonction de sortie

g = A*f' ;

figure(2) ;
subplot(122) ;
plot(x,g) ;
title('signal de sortie') ;
pause

% Inversion 

f_rec = inv(A)*g ; 

figure(3) ;
plot(x,f,x,f_rec) ;
title('signal obtenu par deconvolution directe') ;

err_sol = norm(f-f_rec')/norm(f)*100 

pause

%%% REGULARISATION DE TIKONOV-PHILLIPS %%%

epsilon = 0.0000001

B_eps = inv(A'*A+epsilon*eye(n,n))*A' ; % pseudo inverse de Tikonov-Phillips

cond_B_eps = cond(B_eps)
 
f_eps = B_eps*g ; 
figure(4) ;
plot(x,f,x,f_eps) ;
title('signal obtenu par deconvolution regularisee') ;

err_sol = norm(f-f_eps')/norm(f)*100 









% 
% % perturbations de l'entree
% 
% Ne = 30 ;
% Ampl = .1 ; % amplitude
% for freq = 1:Ne
%     
%     f = f0 + Ampl*sin(2*pi*freq*x) ; % ajout perturbation 
%     
%     figure(4) ;
%     subplot(221) ;
%     plot(x,f0) ; 
%     title('entree d''origine') ; 
%     
%     b = A*f' ;
%     
%     subplot(222) ;
%     plot(x,b0) ; 
%     title('sortie d''origine') ; 
%     
%     subplot(223) ;
%     plot(x,f) ; 
%     title('entree perturbee') ; 
%     
%     subplot(224) ;
%     plot(x,b) ; 
%     title('sortie perturbee') ; 
%     pause
%     
% end
% 
% 
% 
% % reconstruction avec perturbation
% 
% fprintf('reconstruction faussee\n') ;
% 
% f_rec = A\b ;
% 
% figure(5) ;
% subplot(221) ;
% plot(x,b0) ;
% title('donnee fiable') ;
% subplot(223) ;
% plot(x,b) ;
% title('donnee erronee') ;
% subplot(222) ;
% plot(x,f0) ; 
% title('solution esperee') ; 
% subplot(224) ;
% plot(x,f_rec) ; 
% title('solution trouvée') ;
% 
% 
% err_donnee = norm(b-b0)/norm(b)*100 
% 
% err_sol = norm(f0-f_rec')/norm(f0)*100 
% 
% 
% 
