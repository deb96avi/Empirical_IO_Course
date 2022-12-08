% ************************************************************* %
% File name: problem set 1
% Author(s): Daniel Herrera-Araujo - Paul Emile Bernard
% Date: 2022
% Description: Provides solutions to PS1 *
% * * *
% Inputs: imports-85.cvs and also internally generated. 
% ************************************************************* %

clear
clc

%% Exercice #1 %%

% Import dataset : imports-85.cvs using in-house matlab import commands and
% then save it to imports.mat

% Upload the imports.mat
opts=detectImportOptions('imports-85.csv');
opts.VariableNamesLine = 1;
imports=readtable('imports-85.csv',opts, 'ReadVariableNames', true);
save("imports.mat","imports");
load("imports.mat", "imports");

% Create a X, a Nx6 matrix with length, curb-weight, engine-size, horsepower, 
% city-mpg and price as colummns. Use the grp2idx command to transform
% categorical variables into numerical ones. Do not forget to include a
% constant.

colNames = {'length', 'curb_weight', 'engine_size', 'horsepower_peak_rpm', 'city_mpg'};

X = [imports(:,colNames)]; 
X.constant = repmat(1, size(X,1),1);
nums = grp2idx(categorical(X.engine_size));
X.engine_size = nums;


% From the new database remove NAN with rmmissing() command.

X_nan_rm = rmmissing(X);

% Suppose that we only have access to a pricing variable with reduced 
% information. Construct a dichotomous pricing variable equal to one if the
% price is above the median, and zero otherwise. You will need to make sure
% that the price and covariates have the same dimension 

y = [imports(:,'price')];
y.median_price = repmat(1, size(y,1), 1); % dummy for creation

y.median_price(y.price > median(y.price)) = 1;
y.median_price(y.price <= median(y.price)) = 0;
y= removevars(y,{'price'});

% Below condition checks whether covariates and prices have same dimension
height(X_nan_rm)==height(y);
% It doesnt match, so we remove those values from prices where covariate is
% nan
X_ensemble=[X,y];
X_ensemble = rmmissing(X_ensemble);
X=removevars(X_ensemble,{'median_price'});
y=X_ensemble.median_price;

height(X)==height(y);
%% #Reg1 Estimate an OLS model using the analytical formula. 
% The estimation should include the standard deviation, and t-statistics 
% along with an interpretation of each. 
N =  height(X);
X=table2array(X);
b_ols=inv(transpose(X)*X)*transpose(X)*y;

err = y- X*b_ols;

sig2 =  sum(err)^2;
var =  diag(transpose(err)*err*inv(transpose(X)*X)/(N-width(X)));
sd_ols =  sqrt(var);
t_stat =  b_ols./sd_ols;
ols_results = [b_ols sd_ols t_stat];

%% #Reg2 estimate the model using by minimizing the sum of squared residuals 
options = optimoptions('fminunc',...
    'TolFun',1e-12,...
    'MaxFunEvals',1e10,...
    'MaxIter',1e10,...
    'TolX',1e-12,...
    'Display','iter');

%starting values 
X0 = [1, 1, 1, 1, 1, 1] ;
objfunc =@(x) ols(x,y,X);
[b_ml,fval,exitflag,output,grad1,hessian] = fminunc(objfunc,X0,options);
fprintf('The number of iterations was : %d\n', output.iterations);
fprintf('The best function value found was : %g\n', fval);

%computing the standard errors: a property of ML is that, for a correctly
%specified ML, the inverse of the Hessian equals the var-cor matrix.
invH = inv(hessian);
std_ml1 = diag(invH); 
std_ml2 = sqrt(std_ml1);
b_ml=transpose(b_ml);
t_ml =  b_ml./std_ml2;
ml_results1 = ; 
ml_results2 = [b_ml std_ml2 t_ml] ;


%compare the results from the OLS and ML results. The coefficients are
%identical, but the standard errors are not the same. This comes from the
%fact that the model is not correctly specified.

%% #Reg3 estimate the model using by MLE
%residuals with a correctly specified model 
%starting values 
options = optimoptions('fminunc',...
    'TolFun',1e-12,...
    'MaxFunEvals',1e10,...
    'MaxIter',1e10,...
    'TolX',1e-12,...
    'Display','iter');

X0 = [1, 1, 1, 1, 1, 1, 1] ;
objfunc =@(x) mlnorm(x,y,X);
[b_mlcs,fval,exitflag,output,grad2,hessian] = fminunc(objfunc,X0,options);
fprintf('The number of iterations was : %d\n', output.iterations);
fprintf('The best function value found was : %g\n', fval);

err_sigma_ml=b_mlcs(:,width(b_mlcs));
w1=width(b_mlcs)-1;
b_mlcs=b_mlcs(:,1:w1);
b_mlcs=transpose(b_mlcs);
%computing standard errors 
std_mlcs =  sqrt(diag(inv(hessian)));
std_mlcs=std_mlcs(1:w1,:);
t_mlcs =  b_mlcs./std_mlcs;
mlcs_results = [ b_mlcs std_mlcs t_mlcs];


%comparing results

% The coefficients are same in all 3 cases but standard error is different in 2nd case 
% when we do not optimize for the population standard deviation. When we
% control for that, we get same values for coefficient and std. error for ols and maximum likelihood
% estimtaion.

ols_results;
ml_results2;
mlcs_results;


%% Overlay a histogram of the error terms and the pdf of a normal distribution
% using as standard deviation the one that is estimated using the MLE
% results. You will need to first generate the error term (use
% the coefficients from the MLE. To generate the histogram use
% the histogram() command. Use the fitdist() command to fit a normal 
% distribution to the error term. Please comment the results.  
% Why is linear probability model not suited ?

err =  y- X*b_mlcs;
fitd = fitdist(err,'Normal') ;
x_pdf = [-1:0.01:1] ;
pdf_err = pdf(fitd,x_pdf);

% Set up the figure space using figure
figure

% Add sequentially the histogram and then the fitted normal distrubtion.
% you can use the command line(). 
histogram(err)
hold on
plot(x_pdf,pdf_err)
hold off

histfit(err)

% The main advantage of the LPM is its simplicity: OLS regression is 
% computationally simple and quick. In addition, the coefficients have a 
% direct interpretation as marginal effects. The drawbacks are: 
% Marginal effects do not depend on xi, which may give too large magnitude
% close to boundaries 0 and 1. Predicted probabilities might lie outside 
% the interval [0, 1]. 

% Next we will estimate a logit model. To do so, we suppose that the error
% terms follow a logistic distribution.


%% #Reg4 Estimate the Logit model with fminsearch
% You can proceed by creating a function that calculates  the cumulative 
% for a logitic model distribution function (or just exploit the closed form 
% formula of the cumulative function of a logit. Always remember that fminsearch 
% minimizes, so you need to put a negative sign to the output of the
% fminsearh. 

% It seems to be sensitive to starting point
X0 = transpose(b_ols) ;

options =  optimset('TolFun',1e-12,...
    'MaxFunEvals',1e10,...
    'MaxIter',1e10,...
    'TolX',1e-12,...
    'Display','iter');

objfunc =@(x) loglikelihood(x,y,X);
[theta_sim,fval,exitFlag,output] = fminsearch(objfunc,X0,options);
% 
fprintf('The number of iterations was : %d\n', output.iterations);
fprintf('The best function value found was : %g\n', fval);

 
%% Exercice #2 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc

% In the following exercice we will estimate a multinomial logit. We start
% by generating 3 normally distributed variables. You can use randn() command
% to generate them. Make sure to avoid having a mean of zero for the random
% variables. A line should correpond to a choice occasion, while each
% column is considered to be the same characteristic attributed to a 
% different alternative. Thus, there are 3 alternatives and N choice occasions 
% Then, you need to generate three parameters. The values
% it shoud take = 1, -1 and 0.5. We will aim to back out these
% coefficients. 

N = 1000; 
X = normrnd(2,1,[N,3]); 
theta =  transpose([-1 1 0]);

% Next, generate 3 random variables from extreme value distribution using
% the command evrnd()

epsi = evrnd(2,1,[N,3]);

% Create three columns combining the observed part, X*theta, with the 
% unobserved epsi for each of the three alternatives. 

u = X*theta+epsi ;

%% Next, we will find the multinomial outcomes. The intuition of a logit 
% consists on choosing the alternative that provides the largest benefit.
% That is, if there are three alternatives, then if an individual picks
% alternative #1 is because the sum of observed and unobserved parts derived
% by alternative #1 are higher than those from the other two alternatives.

y = zeros(N,1);

for i = 1:N
[a,index]= max([u(i,1) u(i,2) u(i,3)])
    if index==1
        y(i) =  1
    elseif index==2 
        y(i) =  2
    elseif index==3
        y(i) =  3
    end
end

% Count the number of times each alternative is selected and store it into
% n1, n2 and n3. 

n1 = sum(y==1);
n2 = sum(y==2);
n3 = sum(y==3);
n4 = sum(y==0);

save("sml_fake_data.mat","u");
%% Next, we will create an Simulated Maximum likehood estimator of theta 
% To do so, follow the next steps: 
% First, generate 3 vectors d1, d2 and d3 each equally dividing the space. 
% The will capture possible values for parameters theta1, theta2 and theta3 
% assuming that θ1 ∈ [-2, 2]; 
% θ2 ∈ [-2, 2] and θ3 ∈ [-2, 2]; 

e1 = [-1.5:0.05:-0.01] ; 
e2 = [0.01:0.05:1.5] ; 
e3 = 0.5 ;

% We will generate 100 simulated probabilities. Define ns as the number of
% simluations, and create ns sets of Nx3 extreme value random values
ns = 100;
epsi_sim = evrnd(0,1,[ns,N,3]) ;
% 1st simulation matrix: epsi_sim(1,:,:)

%%
% Next, for each combination of (θ1, θ2, θ3) simulate the probability
% of selecting each alternative. Average over the simulations to
% get a precise estimation of each probability at (θ1, θ2, θ3). Then,
% evaluate the loglikelihood at (θ1, θ2, 0.5) using the average probabilities
% and store it. Repeat for all combinations of (θ1, θ2, θ3).  

% Create a loop over each combination of (θ1, θ2, θ3) points
n_t1= width(e1); 
n_t2= width(e2);
n_t3= width(e3);

max_matrix=zeros(n_t1,n_t2);
for i = 1:n_t1 
    for j = 1:n_t2 
        for h = 1:n_t3

            %For each combination of (θ1, θ2, θ3) simulate ns
            %probabilities
            f1=0,f2=0,f3=0;
            for sim = 1:ns
                u = X*[e1(i);e2(j);e3(h)]+squeeze(epsi_sim(sim,:,:));
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
            f1=f1/100;
            f2=f2/100;
            f3=f3/100;
            
            max_matrix(i,j)=n1*log(f1)+n2*log(f2)+n3*log(f3);
            
            % Then, average over simulated probabilities
 
            
            % Compute the ll at the (θ1, θ2, θ3) and store it

             
        end
        

    end
    
end


maximum = max(max(max_matrix));
[theta1_max,theta2_max]=find(max_matrix==maximum);
% Once all the values of the loglikehood are stored, select from the matrix
% the one that maximizes the loglikehood. 

% max of the max of the rows
 
%% Next we provide a graphical representation of the simulated log-likelihood 
% To do it, you will need to use a combination of commands meshgrid, mesh,
% and surf. 

[th1,th2]=meshgrid(e1,e2);

subplot(1,1,1); mesh(th1,th2,llfirst(:,:,11)); 

surf(th1,th2,llfirst(:,:,15));
title('SLL Function: 100 simulations'); xlabel('theta1'); ylabel('theta2'); 
zlabel('simulated log-likelihood'); grid;
hold on

% Please discuss what would happen it we modified the number of simulations
% upward or downward. 


%% Finally, find the estimates using an automatic procedure. Please use 
% fminseach as algorithm. 

%%%%%%%%%%%% This minimizes a discrete function %%%%%%%%%
options =  optimset('TolFun',1e-12,...
    'MaxFunEvals',1e10,...
    'MaxIter',1e10,...
    'TolX',1e-12,...
    'Display','off') ;
for s = 1:10 
    x0 = rand(3,1);
    objfunc =@(x) sml_logit(x,n1,n2,n3,X,epsi_sim);
    [theta_sim,fval,exitFlag,output] = fminsearch(objfunc,x0,options);
    fprintf('The number of iterations was : %d\n', output.iterations);
    fprintf('The best function value found was : %g\n', fval);
    theta_store(s,:) = theta_sim;
    fval_store(s,:) = fval;
end
