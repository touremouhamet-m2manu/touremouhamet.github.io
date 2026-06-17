function compar_eulerien_lagrangien(pb,flx_type,nc,cfl)

close all

Euler_1D(pb,flx_type,nc,cfl,1)
hold on
Lagrange_1D(pb,flx_type,nc,cfl,0)


fig=figure(1);
xlabel('x');
ylabel('densité');
title("Densité à l'instant final");
legend('solution exacte','schéma Eulérien','schéma Lagrangien','Location','northeast');

fig2=figure(2);
xlabel('x');
ylabel('énergie interne');
title("Énergie interne à l'instant final");
legend('solution exacte','schéma Eulérien','Location','northwest');

end