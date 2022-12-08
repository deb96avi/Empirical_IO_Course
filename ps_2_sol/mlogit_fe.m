function [F] = mlogit_fe(theta,D,X1,X2)

    global cdindex cdid
    
    %****************************************%
    %%% Nested Fixed Point Algorithm %%%
    %****************************************% 
    theta_o = rand(size(X2,2),1);
    
    tol = 1;
    while tol > 1e-16

        %****************************************%
        %%% Constructing the logit probability %%%
        %****************************************%       
        delta_out = X1*theta;    
        delta_in  = X2*theta_o;
        expdelta = exp(delta_in + delta_out);
        temp = cumsum(expdelta);
        sum1 = temp(cdindex,:);
        sum1(2:size(sum1,1),:) = diff(sum1);
        sum2 = sum1(cdid,:);
        P = expdelta ./ sum2;   

        %****************************************%
        %%% First order condition %%%
        %****************************************%   

        grad_a = zeros(size(X2, 2),1);
        grad_b = zeros(size(X2, 2),1);

        for kk = 1:size(X2, 2)
            tmp = cumsum(X2(:, kk).* P);
            sum1 = tmp(cdindex, :);
            sum1(2:size(sum1,1), :) = diff(sum1);
            sum2 = sum1(cdid, :);
            deriv_betas1 = (- sum2);
            deriv_betas2 = (X2(:, kk));
            clear tmp sum1 sum2

            tmpa = deriv_betas1(D==1, :);
            tmpb = deriv_betas2(D==1, :);
            grad_a(kk,:) = -sum(tmpa);
            grad_b(kk,:) = sum(tmpb);
        end         

        theta_n = theta_o - log(grad_a./grad_b);
        tol = sum((theta_n-theta_o).^2);
        theta_o = theta_n;
    end

    %****************************************%
    %%% Log likelihood %%%
    %****************************************%

    PP=P(D==1,:);
    F =  -sum(log(PP));
 
end