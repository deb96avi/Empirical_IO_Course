%% creating the minimizing function 

function [v] = gg_ols(beta,y,X)

%compute the residuals 
err = y - X * beta;

%Compute the variance of the score function
g = - 2 * X .* err ;
v = g' * g;
   
end