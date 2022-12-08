function [objf] = loglik_be92(theta,y,x1,x2,x3,tpop,epsilon,sim)


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This function constructs the log-likelihood as in Berry (1992)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %define the size of the matrices containing the observable
    %charactistics for each section of the profit function
    n_x1 = size(x1,2);
    n_x2 = size(x2,2);
    n_x3 = size(x3,2);
   
    %construct the population size. Normalize the coefficient of the
    %population to 1. 
    s = ??????;;
    
    %construct the entry effects on the per-capita demand
    %the coefficient of the 4th entrant to zero.
    alpha1 = ??????;
    alpha2 = ??????;
    alpha3 = ??????;
    alpha4 = 0;
    alpha5 = ??????;

    %construct the entry effects on the fixed-costs  
    gamma1 = ??????;
    gamma2 = ??????;
    gamma3 = ??????;
    gamma4 = ??????;
    gamma5 = ??????;
    
    %construct the per-capita demand 
    v = x2 * theta(n_x1 +1 : n_x1 +n_x2);    
    
    %construct the fixed-costs 
    f = theta(end) * x3;
    
 
%*************************************************%    
%********* Respect the order of entry  ***********%
%*************************************************%

    n_f = 4;
    entry = zeros(size(v,1),n_f,sim); 
    n_star_a = zeros(size(x1,1),sim * 1000); 
    n_star_b = zeros(size(x1,1),sim * 1000); 
    
    
    %%% Start simulation %%%
    for kk = 1:(sim * 1000)
        
        % First, need to check which firms are going to be interested in
        % entering the market. Here, for each firm we need to check
        % whether, given the condition of the markets, they will be
        % prepared to enter or not.
        epsi = epsilon(:,n_f*(kk-1)+1:n_f*(kk));
        for j = 1:n_f
            pi(:,1,j) = (FILL HERE USING YOUR SOLUTION  + epsi(:,j)) > 0;
            pi(:,2,j) = (FILL HERE USING YOUR SOLUTION  + epsi(:,j)) > 0;
            pi(:,3,j) = (FILL HERE USING YOUR SOLUTION  + epsi(:,j)) > 0;
            pi(:,4,j) = (FILL HERE USING YOUR SOLUTION  + epsi(:,j)) > 0;
        end

        %Re-order the firms by profitability. 
        
        
        % 1th approach: 
        % Determine how many firms would be ready to enter market m. 
        % Here we also assume that in each market, potential entrant 1
        % moves first, then potential entrant 2, then 
        % potential entrant 3 and finally the 4th is allowed to enter. 
        n_sub = zeros(size(v,1),n_f);
        
        % Here you can use a while loop!!!!
        
        % Take the max number of firms that enter. 
        n_star_a(:,kk) = max(n_sub,[],2);
  
       
    end
    %%% End simulation %%%
    
    %We match the actual number of firms in the market.
    %by minimizing the squared difference
    
    %Objective function #1 
    objf = sum((...........).^2);   
f=()^2
end

