function [F,grad] = mlogit_fe(theta,D,X1)

    global cdindex cdid
    
    %****************************************%
    %%% Nested Fixed Point Algorithm %%%
    %****************************************% 
    theta_old = theta(1:9);
    
    tol = 1;
    while tol > 1e-16

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
            % extracting sigma from parameteres and multiply separately
            nu_i=alpha_i;
            sigma=theta(size(theta,1));
            theta1=theta(1:size(theta,1)-1);
            delta = X1*theta1; 
            num=exp(delta).*exp(sigma*nu_i.*X1(:,size(X1,2)));
            % summing over choices in denom. and reshaping so that numerator and
            % denominator can be divided
            test1= reshape(num,[10,8192]);
            test2=sum(test1);
            test3=repmat(test2,10,1);
            denom=reshape(test3,[81920,1]);
            P=num./denom;
            sum1=sum1+P;
        end
        sum1=sum1/ns
        
        %****************************************%
        %%% First order condition %%%
        %****************************************%   
        
        % Re-express the FOC in terms of grad_a and grad_b
        %grad_a = zeros(size(X2, 2),1);
        %grad_b = zeros(size(X2, 2),1);

        %for kk = 1:size(X2, 2)
            %You need to construct the FOC for each delta_k.
        %end         
        
        % summing over indiv*time level for each choice j
        sum2=reshape(sum1,[10,8192]);
        sum3=sum(sum2,2);
        
        theta_new = theta_old - sum3(1:9) ;
        tol = sum((theta_new-theta_old).^2);
        theta_old = theta_new;
    end

    %****************************************%
    %%% Log likelihood %%%
    %****************************************%

    F =  -sum(D.*log(sum1));
 
   
end