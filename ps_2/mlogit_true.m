function [F,grad] = mlogit(theta,D,X1)

    global cdindex cdid
    
    %****************************************%
    %%% Constructing the logit probability %%%
    %****************************************%
    ns=100;
    sum1=0;
    for i=1:ns
        % creating random variations in preferences for each of the 820 individuals
        alpha_i1=theta(10)+randn(1,819)*theta(11);
        alpha_i2=theta(10)+randn(1,1)*theta(11);

        alpha_i1=repmat(alpha_i1,100,1);
        alpha_i2=repmat(alpha_i2,20,1);
        alpha_i1=reshape(alpha_i1,[81900,1]);
        alpha_i2=reshape(alpha_i2,[20,1]);

        alpha_i=[alpha_i1;alpha_i2];
        % numerator depends on choice but not denominator
        exp_price=exp(X1(:,10).*alpha_i)
        delta = X1(:,1:9)*theta(1:9);    
        expdelta =  exp(delta).*exp_price;
        % summing over choices in denom. and reshaping so that numerator and
        % denominator can be divided
        test1= reshape(expdelta,[10,8192]);
        test2=sum(test1);
        test3=repmat(test2,10,1);
        test3=reshape(test3,[81920,1]);
        %P containts the probability for each individual and choice occasion.
        %The size P is 81920x1. 
        P =  expdelta./test3;   
        sum1=sum1+P;
    end
    sum1=sum1/ns
    %****************************************%
    %%% Log likelihood %%%
    %****************************************%

    F =  -sum(D.*log(sum1));
 
end