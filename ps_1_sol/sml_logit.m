%% creating the minimizing function 

function [f] = sml_logit(theta,n1,n2,n3,X,epsi_sim)

            %For each combination of (θ1, θ2, 0.5) simulate ns
            %probabilities 
            N = size(epsi_sim,1);
            ns = size(epsi_sim,3);
            for sim = 1:ns
                u_s = [X(:,1) * theta(1), X(:,2) * theta(2), X(:,3) * 0] + epsi_sim(:,:,sim);
                x1(u_s(:,1) > u_s(:,2) & u_s(:,1) > u_s(:,3)) = 1;
                x2(u_s(:,2) > u_s(:,1) & u_s(:,2) > u_s(:,3)) = 1; 
                x3(u_s(:,3) > u_s(:,1) & u_s(:,3) > u_s(:,2)) = 1; 
                ps1(sim)=sum(x1)/N;
                ps2(sim)=sum(x2)/N;
                ps3(sim)=sum(x3)/N;
                x1 = zeros(N,1);
                x2 = zeros(N,1);
                x3 = zeros(N,1);      
            end 
            
            % Then, average over simulated probabilities
            p1 = sum(ps1) / ns;
            p2 = sum(ps2) / ns;
            p3 = sum(ps3) / ns;
            
            % Compute the ll at the (θ1, θ2, 0.5) and store it
           f = -(n1.*log(p1) + n2.*log(p2) + n3.*log(p3));
end