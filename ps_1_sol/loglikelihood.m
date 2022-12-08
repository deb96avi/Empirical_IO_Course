%% creating the minimizing function 

function [f] = loglikelihood(theta,X,y)

    % Computes the exp 
    expX = exp( X*theta );

    % Compute the probabilities 
    p1 = expX./(1+expX);

    % Set the loglikelihood 
    llsecond = y.*log(p1) + (1-y).*log(1-p1); 
    f = -sum(llsecond);   
   
end
