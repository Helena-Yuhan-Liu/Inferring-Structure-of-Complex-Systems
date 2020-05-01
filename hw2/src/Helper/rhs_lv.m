function rhs=rhs_lv(t,x,dummy,b,p,r,d)
% An ODE function of the lotka-volterra model to be 
% solved by ode45()

% Lotka-volterra model: 
rhs=[(b-p*x(2))*x(1); (r*x(1)-d)*x(2)];