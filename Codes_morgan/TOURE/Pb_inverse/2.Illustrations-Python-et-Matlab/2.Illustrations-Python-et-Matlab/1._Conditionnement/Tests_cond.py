#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#from math import *
from numpy import *
#from scipy import * 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("",end='\n\n\n') 
print("Tests de conditionnements", end='\n\n')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("----------- Matrice de Wilson --------------", end='\n\n') ;  
A = array ([[10, 7, 8, 7], [7 ,5, 6, 5], [8, 6, 10, 9], [7, 5, 9, 10]])
print(A)
#print(A.shape)
input() 

detA = linalg.det(A) 
print("Déterminant de A = ", detA, end='\n\n')
input() 

A_inv = linalg.inv(A)
print("Inverse de A", end='\n\n')
print(A_inv, end='\n\n')
input() 

cond1 = linalg.cond(A,1)
condinf = linalg.cond(A,inf)
cond2 = linalg.cond(A)
print("Conditionnement en norme 2 : ", cond2, end='\n\n')
input() 

vp,Vp = linalg.eig(A)
print("Valeurs propres : ", vp, end='\n\n')
input() 

b = array([[32], [23], [33], [31]])
print("Donnée =", transpose(b), end='\n\n')
#print(b.shape)
input() 

x = dot(linalg.inv(A),b)
print("Solution de Ax=b : x = ", transpose(x))
input() 

b_err = b + array([[0.1], [-0.1], [0.1], [-0.1]])
print("Donnée perturbée =", transpose(b_err), end='\n\n')
erreur_donnee = linalg.norm(b-b_err)/linalg.norm(b)*100 ; 
print("Erreur sur la donnée b + db = {: .2f} % ".format(erreur_donnee), end='\n\n') ; 
input() 

x_err=linalg.solve(A, b_err)
print("Solution de Ax=b perturbé : x + dx = ", transpose(x_err), end='\n\n')
erreur_sol = linalg.norm(x-x_err)/linalg.norm(x)*100 
print("Erreur sur la solution = {: .2f} %".format(erreur_sol), end='\n\n') ; 
input() 

input("...") ; print("",end='\n\n') 


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("------------ Matrices de Hilbert -----------", end='\n\n')

for n in range(1,21):
    b = zeros(n)  
    H = zeros((n,n))
    for i in range(0,n):
        S = 0 
        for j in range(0,n):
            H[i,j] = 1/(i+j+1) 
            S = S + H[i,j] 
        b[i] = S
    cond2 = linalg.cond(H) 
    print("Le conditionnement en norme 2 pour n = {} vaut :  {: .2e}".format(n,cond2)) 
    x = linalg.solve(H, b)
    print("La solution de Hx = b est x = ", transpose(x))
    input()