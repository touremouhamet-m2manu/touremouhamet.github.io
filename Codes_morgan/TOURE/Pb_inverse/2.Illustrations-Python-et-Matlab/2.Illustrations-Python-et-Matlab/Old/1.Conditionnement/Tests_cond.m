
clear all ; close all ; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n\n\n') ;  
fprintf('\n *********** Matrice de Wilson ************\n') ;

A = [ 10 7 8  7 ;
      7  5 6  5 ;
      8 6  10 9 ;
      7 5  9 10]

  
pause
invA = inv(A) 

pause
detA = det(A) 

pause
cond1 = cond(A,1)
condinf = cond(A,inf)
cond2 = cond(A)

pause
vp = eig(A)

pause
b = [32,23,33,31]'
x = A\b

pause
b_err = b + [0.1,-0.1,0.1,-0.1]'
erreur_donnee = norm(b-b_err)/norm(b)*100 ; 
fprintf('Erreur sur la donnée = %2.2f %%\n',erreur_donnee) ; 
pause
x_err = A\b_err
erreur_sol = norm(x-x_err)/norm(x)*100 ;
fprintf('Erreur sur la solution = %2.2f %%\n',erreur_sol) ; 

pause
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n\n\n') ;  
fprintf('*********  Matrices de Hilbert **********\n') ;

for n = 2:30
   b = zeros(n,1) ; 
   for i = 1:n
   S = 0 ; 
       for j=1:n
           H(i,j) = 1/(i+j-1) ; 
           S = S + H(i,j) ; 
       end
   b(i) = S ; 
   end
   % H
   
   cond2 = cond(H) 
    pause
    x = H\b
    pause
end
%plot(1:9,condv) ; 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fprintf('\n **********  Matrices aleatoires  ***********\n') ;
% 
% %condv = zeros(9,1) ; 
% n = 512 ; 
% for nexp = 1:1000 
%    B = rand(n,n)-1/2*ones(n,n) ; % la grandeur peut etre ramenee à [0,1]... 
%    cond2 = cond(B) 
%    cond2bis = cond(B+7*eye(n,n))
%    pause
% end
% %plot(1:9,condv) ; 
% 
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Matrice de derivation
% n = 1000 ;
% 
% u = -1*ones(1,n-1) ;
% v = ones(1,n) ;
% InvA = diag(u,-1) + diag(v) ; 
% A = inv(InvA) ; 
% 
% cond(A,1) 
% 
% eps = 0.01 ; 
% 
% Atp = inv(A'*A + eps*eye(n))*A' ;
% 
% cond(Atp,1) 