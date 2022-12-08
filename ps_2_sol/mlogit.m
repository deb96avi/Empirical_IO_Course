function [F] = mlogit(theta,D,X1)

    global cdindex cdid
    
    %****************************************%
    %%% Constructing the logit probability %%%
    %****************************************%

    delta = X1*theta;    
    expdelta = exp(delta) ;
    temp = cumsum(expdelta);
    sum1 = temp(cdindex,:);
    sum1(2:size(sum1,1),:) = diff(sum1);
    sum2 = sum1(cdid,:);
    P = expdelta ./ sum2;   

    %****************************************%
    %%% Log likelihood %%%
    %****************************************%

    PP=P(D==1,:);
    F =  -sum(log(PP));
 
end