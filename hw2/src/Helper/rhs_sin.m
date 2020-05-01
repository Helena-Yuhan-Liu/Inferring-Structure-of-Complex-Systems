function rhs=rhs_sin(t,x,dummy,xi,yi,indx,indy)
% An ODE function of the sinusoidal library to be 
% solved by ode45()

xs=x(1); ys=x(2); 

% Build the library 
A=[xs ys]; 
for om=0.2:0.5:5
    A=[A sin(om*t) cos(om*t)];
end 
Ax=A; Ax(:,indx)=[]; Ax=Ax';
Ay=A; Ay(:,indy)=[]; Ay=Ay'; 

% The dynamical system x_dot=Ax
rhs=[xi'*Ax; yi'*Ay];
