function [f] = sml_logit(theta,n1,n2,n3,X,epsi_sim)
            %For each combination of (θ1, θ2, 0.5) simulate ns
            %probabilities 
            N = size(epsi_sim,2);
            ns = size(epsi_sim,1);
            theta_vec=[theta(1);theta(2);theta(3)];
            f1=0,f2=0,f3=0;
            for sim = 1:ns
                u = X*theta_vec+squeeze(epsi_sim(sim,:,:));
                y = zeros(N,1);
                for k = 1:N
                [a,index]= max([u(k,1) u(k,2) u(k,3)]);
                    if index==1
                        y(k) =  1;
                    elseif index==2 
                        y(k) =  2;
                    elseif index==3
                        y(k) =  3;
                    end
                end
                n1_1 = sum(y==1);
                n2_1 = sum(y==2);
                n3_1 = sum(y==3);
                
                f1=f1+n1_1/N;
                f2=f2+n2_1/N;
                f3=f3+n3_1/N;
            end 
            
            % Then, average over simulated probabilities
            f1=f1/100;
            f2=f2/100;
            f3=f3/100;
            
            % Compute the ll at the (θ1, θ2, 0.5) and store it
           f = -(n1*log(f1) + n2*log(f2) + n3*log(f3));
end
