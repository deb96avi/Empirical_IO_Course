function [F,grad] = mlogit(theta,D,X1)

    global cdindex cdid
    
    %****************************************%
    %%% Constructing the logit probability %%%
    %****************************************%

    delta = X1*theta;    
    expdelta =  exp(delta);
    test1= reshape(expdelta,[10,8192]);
    test2=sum(test1);
    test3=repmat(test2,10,1);
    test3=reshape(test3,[81920,1]);
    %P containts the probability for each individual and choice occasion.
    %The size P is 81920x1. 
    P =  expdelta./test3;   

    %****************************************%
    %%% Log likelihood %%%
    %****************************************%

    F =  -sum(D.*log(P));
 
end