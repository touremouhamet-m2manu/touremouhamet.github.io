function Euler_1D(pb,flx_type,nc,cfl,plt)

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
mom=zeros(nc,1);
ene=zeros(nc,1);
cel=zeros(nc,1);
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
    
    mom(ii)=den(ii)*vit(ii);
    ene(ii)=pre(ii)/(gam-1)+den(ii)*vit(ii)^2/2;
    cel(ii)=sqrt(pre(ii)*gam/den(ii));

    Vsol(ii,1)=den(ii);
    Vsol(ii,2)=mom(ii);
    Vsol(ii,3)=ene(ii);
end

tps=0;

flux_int=zeros(nc+1,3);

n_it=0;
while (tps<tfin)
    
    dt=cfl*dx/max(cel+abs(vit));
    if (tps+dt>tfin+1.D-6)
        dt=tfin-tps;
    end
    
    flux_int(1,:)=flux_num(Vsol(1,:),Vsol(1,:),gam,flx_type);
    for ii=1:nc
        
        if (ii<nc)
            flux_int(ii+1,:)=flux_num(Vsol(ii,:),Vsol(ii+1,:),gam,flx_type);
        else
            flux_int(ii+1,:)=flux_num(Vsol(ii,:),Vsol(ii,:),gam,flx_type);
        end
        
        Vsol(ii,:)=Vsol(ii,:)-dt/dx*(flux_int(ii+1,:)-flux_int(ii,:));
        
        den(ii)=Vsol(ii,1);
        vit(ii)=Vsol(ii,2)/Vsol(ii,1);
        pre(ii)=(gam-1)*(Vsol(ii,3)-0.5*Vsol(ii,2)^2/Vsol(ii,1));
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
    plot(xi,den,'-*b');

    xlabel('x');
    ylabel('densité');
    title("Densité à l'instant final");
    legend('solution exacte','schéma Eulérien','Location','northeast');

    fig2=figure(2);
    rect=[1000 -100 800 600];
    set(fig2,'Position',rect);
    plot(xs,eni_sol,'-r','linewidth',2);
    hold on
    plot(xi,pre./((gam-1)*den),'-*b');

    xlabel('x');
    ylabel('énergie interne');
    title("Énergie interne à l'instant final");
    legend('solution exacte','schéma Eulérien','Location','northwest');
else
    fig1=figure(1);
    plot(xi,den,'-*b');

    fig2=figure(2);
    plot(xi,pre./((gam-1)*den),'-*b');
end

end


function FF=flux_num(VL,VR,gam,flx_type)
        
den_L=VL(1);
vit_L=VL(2)/VL(1);
pre_L=(gam-1)*(VL(3)-0.5*VL(2)^2/VL(1));
cel_L=sqrt(pre_L*gam/den_L);

den_R=VR(1);
vit_R=VR(2)/VR(1);
pre_R=(gam-1)*(VR(3)-0.5*VR(2)^2/VR(1));
cel_R=sqrt(pre_R*gam/den_R);

FL(1)=VL(2);
FL(2)=den_L*vit_L^2+pre_L;
FL(3)=(VL(3)+pre_L)*vit_L;

FR(1)=VR(2);
FR(2)=den_R*vit_R^2+pre_R;
FR(3)=(VR(3)+pre_R)*vit_R;


if (flx_type==1)
    S0=max(cel_L+abs(vit_L),cel_R+abs(vit_R));
    
    FF=0.5*(FL+FR-S0*(VR-VL));
elseif (flx_type==2)
    %SL=vit_L-cel_L;
    %SR=vit_R+cel_R;
    SL=min(vit_L-cel_L,vit_R-cel_R);
    SR=max(vit_L+cel_L,vit_R+cel_R);
    
    if (SL>=0)
        FF=FL;
    elseif (SL<=0 && SR>=0)
        FF=(SR*FL-SL*FR+SL*SR*(VR-VL))/(SR-SL);
    else
        FF=FR;
    end
else
    %SL=vit_L-cel_L;
    %SR=vit_R+cel_R;
    SL=min(vit_L-cel_L,vit_R-cel_R);
    SR=max(vit_L+cel_L,vit_R+cel_R);
    
    S0=den_R*vit_R*(SR-vit_R)-den_L*vit_L*(SL-vit_L)-pre_R+pre_L;
    S0=S0/(den_R*(SR-vit_R)-den_L*(SL-vit_L));
    
    VVL(1)=den_L*(SL-vit_L)/(SL-S0);
    VVL(2)=VVL(1)*S0;
    VVL(3)=VVL(1)*(VL(3)/den_L+(S0-vit_L)*(S0+pre_L/(den_L*(SL-vit_L))));
    
    VVR(1)=den_R*(SR-vit_R)/(SR-S0);
    VVR(2)=VVR(1)*S0;
    VVR(3)=VVR(1)*(VR(3)/den_R+(S0-vit_R)*(S0+pre_R/(den_R*(SR-vit_R))));
    
    if (SL>=0)
        FF=FL;
    elseif (SL<=0 && S0>=0)
        FF=FL+SL*(VVL-VL);
    elseif (S0<=0 && SR>=0)
        FF=FR+SR*(VVR-VR);
    else
        FF=FR;
    end
end

end