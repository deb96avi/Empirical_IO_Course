%% creating the minimizing function 

function [f] = ols(theta,y,X)

%compute the residuals 
err = y - X * theta;

%Add the squared residuals
f = sum(err.^2);
   
end
