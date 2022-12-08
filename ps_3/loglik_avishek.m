function [f] = loglik_avishek(theta,X,ns)

    S=X{:,'tpop'}+theta(1)*X{:,'opop'}+theta(2)*X{:,'ngrw'}+theta(3)*X{:,'pgrw'}+theta(4)*X{:,'octy'};
    V=theta(5)*X{:,'eld'}+theta(6)*X{:,'pinc'}+theta(7)*X{:,'lnhdd'}+theta(8)*X{:,'ffrac'};
    F=theta(9)*X{:,'landv'};

    n_star_avg=zeros(size(X,1),1);

    index_table=[0*ones(1,size(X,1));1*ones(1,size(X,1));2*ones(1,size(X,1));3*ones(1,size(X,1));4*ones(1,size(X,1))]';

    for s=1:ns
        n_max_table=zeros(size(X,1),5);
        eps_sim=normrnd(0,1,[size(X,1),4]);
        % n_min: no. of firms already in market when entrant takes decision
        for n_min=0:4
            profit_im=zeros(size(X,1),4);
            for i=1:4
                profit_im(:,i)=S.*(theta(n_min+10)+V)-F-theta(n_min+14)+eps_sim(:,i);    
            end
            profit_im2=profit_im(:,:)>=0;
            % no. of firms entering when 'n_min' firms present in market
            n_max_table(:,n_min+1)=sum(profit_im2(:,:),2);

        end
        
        index_highest=n_max_table>=index_table;
        n_star=zeros(size(X,1),1);
        for j=1:size(X,1)
            [t1,t2]=find(index_highest(j,:)==max(index_highest(j,:)),1,'last');
            n_star(j,1)= t2;
        end
        n_star=n_star-1;
        n_star_avg=n_star_avg+n_star;
    end

    n_star_avg=n_star_avg/ns;

    f=sum(n_star_avg-X{:,'tire'})^2;
end

