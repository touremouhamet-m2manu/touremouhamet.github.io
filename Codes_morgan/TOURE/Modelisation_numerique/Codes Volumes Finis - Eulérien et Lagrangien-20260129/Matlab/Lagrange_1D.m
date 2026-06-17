function Lagrange_1D(pb,flx_type,nc,cfl,plt)

if (plt==1)
    close all;
end

if (pb==1)
    xd=0;
    xf=1;
    xdis=0.5;
    gam=1.4;
    tfin=0.2;
    den_L=1;
    pre_L=1;
    vit_L=0;
    den_R=0.125;
    pre_R=0.1;
    vit_R=0;
elseif (pb==2)
    xd=0;
    xf=1;
    xdis=0.5;
    gam=1.4;
    tfin=0.15;
    den_L=1;
    pre_L=0.4;
    vit_L=-2;
    den_R=1;
    pre_R=0.4;
    vit_R=2;
end

dx=(xf-xd)/nc;
xint=linspace(xd,xf,nc+1);
xi=xd+dx/2:dx:xf-dx/2;

den=zeros(nc,1);
vit=zeros(nc,1);
pre=zeros(nc,1);
ene=zeros(nc,1);
cel=zeros(nc,1);
masse=zeros(nc,1);
Vsol=zeros(nc,3);
for ii=1:nc    
    if (xint(ii+1)<=xdis)
        den(ii)=den_L;
        pre(ii)=pre_L;
        vit(ii)=vit_L;
    else
        den(ii)=den_R;
        pre(ii)=pre_R;
        vit(ii)=vit_R;
    end
    
    ene(ii)=pre(ii)/(den(ii)*(gam-1))+vit(ii)^2/2;
    cel(ii)=sqrt(pre(ii)*gam/den(ii));
    masse(ii)=den(ii)*dx;

    Vsol(ii,1)=1/den(ii);
    Vsol(ii,2)=vit(ii);
    Vsol(ii,3)=ene(ii);
end

tps=0;

flux_int=zeros(nc+1,3);

n_it=0;
while (tps<tfin)   

    dt=cfl*min(masse)/max(den.*cel);
    if (tps+dt>tfin+1.D-6)
        dt=tfin-tps;
    end
    
    flux_int(1,:)=flux_num(Vsol(1,:),Vsol(1,:),gam,flx_type);
    xint(1)=xint(1)-dt*flux_int(1,1);
    for ii=1:nc
        
        if (ii<nc)
            flux_int(ii+1,:)=flux_num(Vsol(ii,:),Vsol(ii+1,:),gam,flx_type);
        else
            flux_int(ii+1,:)=flux_num(Vsol(ii,:),Vsol(ii,:),gam,flx_type);
        end
        
        Vsol(ii,:)=Vsol(ii,:)-dt/masse(ii)*(flux_int(ii+1,:)-flux_int(ii,:));
        xint(ii+1)=xint(ii+1)-dt*flux_int(ii+1,1);
        xi(ii)=(xint(ii)+xint(ii+1))/2;
        
        den(ii)=1/Vsol(ii,1);
        vit(ii)=Vsol(ii,2);
        pre(ii)=den(ii)*(gam-1)*(Vsol(ii,3)-0.5*vit(ii)^2);
        cel(ii)=sqrt(pre(ii)*gam/den(ii));
    end
    
    tps=tps+dt;
    n_it=n_it+1;
    
end

if (pb==1)
    file_den_sol='Sol_exactes/sod_den.dat';
    file_eni_sol='Sol_exactes/sod_eni.dat';
elseif (pb==2)
    file_den_sol='Sol_exactes/dbl_det_den.dat';
    file_eni_sol='Sol_exactes/dbl_det_eni.dat';
end
fid_den = fopen(file_den_sol,'r');
TT = fread(fid_den);
ss=sum(TT==10);
fclose(fid_den);

fid_den = fopen(file_den_sol,'r');
fid_eni = fopen(file_eni_sol,'r');
xs=zeros(1,ss);
den_sol=zeros(1,ss);
eni_sol=zeros(1,ss);
for ii=1:ss   
  a_den=fscanf(fid_den,'%g %g',[2 1]);
  a_eni=fscanf(fid_eni,'%g %g',[2 1]);
  xs(ii)=a_den(1);
  den_sol(ii)=a_den(2);
  eni_sol(ii)=a_eni(2);
end

if (plt==1)
    fig1=figure(1);
    rect=[100 -100 800 600];
    set(fig1,'Position',rect);
    plot(xs,den_sol,'-r','linewidth',2);
    hold on
    plot(xi,den,'-dk');

    xlabel('x');
    ylabel('densité');
    title("Densité à l'instant final");
    legend('solution exacte','schéma Lagrangien','Location','northeast');

    fig2=figure(2);
    rect=[1000 -100 800 600];
    set(fig2,'Position',rect);
    plot(xs,eni_sol,'-r','linewidth',2);
    hold on
    plot(xi,pre./((gam-1)*den),'-dk');

    xlabel('x');
    ylabel('énergie interne');
    title("Énergie interne à l'instant final");
    legend('solution exacte','schéma Lagrangien','Location','northwest');
else
    fig1=figure(1);
    plot(xi,den,'-dk');

    fig2=figure(2);
    plot(xi,pre./((gam-1)*den),'-dk');
end

end


function FF=flux_num(VL,VR,gam,flx_type)
        
den_L=1/VL(1);
vit_L=VL(2);
ene_L=VL(3);
pre_L=den_L*(gam-1)*(VL(3)-0.5*VL(2)^2);
cel_L=sqrt(pre_L*gam/den_L);

den_R=1/VR(1);
vit_R=VR(2);
ene_R=VR(3);
pre_R=den_R*(gam-1)*(VR(3)-0.5*VR(2)^2);
cel_R=sqrt(pre_R*gam/den_R);


if (flx_type==1)
    Z0=max(den_L*cel_L,den_R*cel_R);
    
    vit_riemann=0.5*(vit_L+vit_R+Z0*(1/den_R-1/den_L));
    pre_riemann=0.5*(pre_L+pre_R-Z0*(vit_R-vit_L));
    pre_vit_riemann=0.5*(pre_L*vit_L+pre_R*vit_R-Z0*(ene_R-ene_L));
elseif (flx_type==2)
    Z0=max(den_L*cel_L,den_R*cel_R);
    
    vit_riemann=0.5*(vit_L+vit_R-(pre_R-pre_L)/Z0);
    pre_riemann=0.5*(pre_L+pre_R-Z0*(vit_R-vit_L));
    pre_vit_riemann=pre_riemann*vit_riemann;
else
    ZL=den_L*cel_L;
    ZR=den_R*cel_R;
    
    vit_riemann=(ZL*vit_L+ZR*vit_R-(pre_R-pre_L))/(ZL+ZR);
    pre_riemann=(ZR*pre_L+ZL*pre_R-ZL*ZR*(vit_R-vit_L))/(ZL+ZR);
    pre_vit_riemann=pre_riemann*vit_riemann;
end

FF(1)=-vit_riemann;
FF(2)=pre_riemann;
FF(3)=pre_vit_riemann;

end