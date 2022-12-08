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
load('imports.mat')

% Create a X, a Nx6 matrix with length, curb-weight, engine-size, horsepower, 
% city-mpg and price as colummns. Use the grp2idx command to transform
% categorical variables into numerical ones. Do not forget to include a
% constant.

X = [ ones(size(length1,1),1) length1 curbweight ...
      grp2idx(enginesize) horsepowerpeakrpm citympg]; 

% From the new database remove NAN with rmmissing() command.

X = rmmissing(X); 

% Suppose that we only have access to a pricing variable with reduced 
% information. Construct a dichotomous pricing variable equal to one if the
% price is above the median, and zero otherwise. You will need to make sure
% that the price and covariates have the same dimension 

y = price > median(price);
y = y(isnan(citympg) ~= 1);

%% #Reg1 Estimate an OLS model using the analytical formula. 
% The estimation should include the standard deviation, and t-statistics 
% along with an interpretation of each. 
N = size(y,1);
b_ols = (X' * X)\(X' * y);
err = y - X * b_ols;

sig2 = (err' * err)/(N-2);
var = inv(X' * X) * sig2; 
sd_ols = diag(sqrt(var));
t_stat = b_ols./sd_ols;
ols_results = [b_ols sd_ols t_stat]

%% #Reg2 estimate the model using by minimizing the sum of squared residuals 
options = optimoptions('fminunc',...
    'TolFun',1e-12,...
    'MaxFunEvals',1e10,...
    'MaxIter',1e10,...
    'TolX',1e-12,...
    'Display','iter');

%starting values 
X0 = zeros(size(X,2),1);
objfunc =@(x) ols(x,y,X);
[b_ml,fval,exitflag,output,grad1,hessian] = fminunc(objfunc,X0,options);
fprintf('The number of iterations was : %d\n', output.iterations);
fprintf('The best function value found was : %g\n', fval);

%computing the standard errors: a property of ML is that, for a correctly
%specified ML, the inverse of the Hessian equals the var-cor matrix.
invH = inv(hessian);
std_ml1 = diag(sqrt(invH));
std_ml2 = diag(sqrt(invH * (gg_ols(b_ml,y,X)) * invH));
ml_results1 = [b_ml std_ml1  b_ml./std_ml1]
ml_results2 = [b_ml std_ml2  b_ml./std_ml2]
ols_results = [b_ols sd_ols  b_ols./sd_ols]

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

X0 = ones(size(X,2)+1,1)/10;
objfunc =@(x) mlnorm(x,y,X);
[b_mlcs,fval,exitflag,output,grad2,hessian] = fminunc(objfunc,X0,options);
fprintf('The number of iterations was : %d\n', output.iterations);
fprintf('The best function value found was : %g\n', fval);

%computing standard errors 
std_mlcs = diag(sqrt(inv(hessian)));
mlcs_results = [b_mlcs std_mlcs b_mlcs./std_mlcs];


%comparing results
ols_results
ml_results1
ml_results2
mlcs_results


%% Overlay a histogram of the error terms and the pdf of a normal distribution
% using as standard deviation the one that is estimated using the MLE
% results. You will need to first generate the error term (use
% the coefficients from the MLE. To generate the histogram use
% the histogram() command. Use the fitdist() command to fit a normal 
% distribution to the error term. Please comment the results.  
% Why is linear probability model not suited ?

err = y - X * b_ml;
fitd = fitdist(err,'normal');
x_pdf = [-3:0.1:3];
pdf_err = pdf(fitd,x_pdf);

% Set up the figure space using figure
figure

% Add sequentially the histogram and then the fitted normal distrubtion.
% you can use the command line(). 
histogram(err,'Normalization','pdf')
line(x_pdf,pdf_err)


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

% X0 = zeros(size(X,2),1);
X0 = randn(size(X,2),1)/100;

options =  optimset('TolFun',1e-12,...
    'MaxFunEvals',1e10,...
    'MaxIter',1e10,...
    'TolX',1e-12,...
    'Display','iter');

objfunc =@(x) loglikelihood(x,X,y);
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

N = 10000; 
X = [1+randn(N,1) 2+randn(N,1) 0*randn(N,1) ];
theta = [1; -1; 0];

% Next, generate 3 random variables from extreme value distribution using
% the command evrnd()

epsi = evrnd(0,1,N,3);

% Create three columns combining the observed part, X*theta, with the 
% unobserved epsi for each of the three alternatives. 

u = [X(:,1) * theta(1), X(:,2) * theta(2), X(:,3) * theta(3)] + epsi;

%% Next, we will find the multinomial outcomes. The intuition of a logit 
% consists on choosing the alternative that provides the largest benefit.
% That is, if there are three alternatives, then if an individual picks
% alternative #1 is because the sum of observed and unobserved parts derived
% by alternative #1 are higher than those from the other two alternatives.

y = zeros(N,1);

for i = 1:N
    if u(i,1) > u(i,2) && u(i,1) > u(i,3)
        y(i) = 1;
    elseif u(i,2) > u(i,1) && u(i,2) > u(i,3)
        y(i) = 2;
    elseif u(i,3) > u(i,1) && u(i,3) > u(i,2)
        y(i) = 3;
    end
end

% Count the number of times each alternative is selected and store it into
% n1, n2 and n3. 

n1 = sum(y==1);
n2 = sum(y==2);
n3 = sum(y==3);
n4 = sum(y==0);

%% Next, we will create an Simulated Maximum likehood estimator of theta 
% To do so, follow the next steps: 
% First, generate 3 vectors d1, d2 and d3 each equally dividing the space. 
% The will capture possible values for parameters theta1, theta2 and theta3 
% assuming that θ1 ∈ [-2, 2]; 
% θ2 ∈ [-2, 2] and θ3 ∈ [-2, 2]; 

e1 = [0.8:0.01:1.2]; 
e2 = [-1.2:0.01:-0.8]; 
e3 = [0];

% We will generate 100 simulated probabilities. Define ns as the number of
% simluations, and create ns sets of Nx3 extreme value random values
ns = 10;
epsi_sim = evrnd(0,1,N,3,ns);

%%
% Next, for each combination of (θ1, θ2, θ3) simulate the probability
% of selecting each alternative. Average over the simulations to
% get a precise estimation of each probability at (θ1, θ2, θ3). Then,
% evaluate the loglikelihood at (θ1, θ2, 0.5) using the average probabilities
% and store it. Repeat for all combinations of (θ1, θ2, θ3).  

% Create a loop over each combination of (θ1, θ2, θ3) points
n_t1=length(e1); 
n_t2=length(e2);
n_t3=length(e3);

% create a 
x1 = zeros(N,1);
x2 = zeros(N,1);
x3 = zeros(N,1);

for i = 1:n_t1 
    for j = 1:n_t2 
        for h = 1:n_t3

            %For each combination of (θ1, θ2, θ3) simulate ns
            %probabilities 
            for sim = 1:ns
                u_s = [X(:,1) * e1(i), X(:,2) * e2(j), X(:,3) * e3(h)] + epsi_sim(:,:,sim);
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
            p1(i,j,h) = sum(ps1) / ns;
            p2(i,j,h) = sum(ps2) / ns;
            p3(i,j,h) = sum(ps3) / ns;
            
            % Compute the ll at the (θ1, θ2, θ3) and store it
            llfirst(i,j,h) = n1.*log(p1(i,j,h)) + n2.*log(p2(i,j,h)) + n3.*log(p3(i,j,h));
            clear s1 s2 s3
        end

    end
    
end

% Once all the values of the loglikehood are stored, select from the matrix
% the one that maximizes the loglikehood. 

[ll11,row1]=max(llfirst); % max from the rows
[ll22,col2]=max(ll11);
[ll33,dim3]=max(ll22);

% max of the max of the rows
thetahat1=[e1(row1(col2(dim3))),e2(col2(dim3)), e3(dim3)];

%% Next we provide a graphical representation of the simulated log-likelihood 
% To do it, you will need to use a combination of commands meshgrid, mesh,
% and surf. 

[th1,th2]=meshgrid(e1,e2);

subplot(1,1,1); mesh(th1,th2,llfirst); 

surf(th1,th2,llfirst);
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

