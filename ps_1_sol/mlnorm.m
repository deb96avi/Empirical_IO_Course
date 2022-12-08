%% creating the an ML function to parallel OLS

function [f] = mlnorm(theta,y,X)

%coefficients
beta = theta(1:end-1);
sig = theta(end);

% number of observations
N = size(y,1);

%compute the residuals 
err = y - X * beta;
sig2 = sig^2;

%Add the squared residuals
f = -(-0.5 * N * log( 2*pi ) -0.5 * N * log( sig2 ) - 0.5*( sig2 )^(-1)*(err'*err) ) ;
   
end
