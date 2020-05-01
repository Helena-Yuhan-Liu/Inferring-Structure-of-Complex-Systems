function rhs=rhs_poly(t,x,dummy,xi,yi,indx,indy)
% An ODE function of the 4th order polynomial library to be 
% solved by ode45()

xs=x(1); ys=x(2); 

% Build the library 
A=[xs ys xs.^2 xs.*ys ys.^2 xs.^3 (ys.^2).*xs (xs.^2).*ys ys.^3 ...
    xs.^4 (xs.^3).*ys (xs.^2).*(ys.^2) (ys.^3).*xs ys.^4];
Ax=A; Ax(:,indx)=[]; Ax=Ax';
Ay=A; Ay(:,indy)=[]; Ay=Ay'; 

% The dynamical system x_dot=Ax
rhs=[xi'*Ax; yi'*Ay];