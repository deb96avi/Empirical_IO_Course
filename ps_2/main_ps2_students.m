% ************************************************************* %
% File name: Demand estimation
% Author(s): Daniel Herrera-Araujo
% Date: 2022
% Description: Estimation of MNL logit using MLE and NFP 
% * * *
% Inputs: datasets on demand
% ************************************************************* %

% Bonus Ques ans: NML shows different parameter estimates compared to the
% true values because it doesn't account for consumer heterogeneity in
% preferences. All agents are assumed to respond the same way to prices. 
% Thus the variance (sigma) is zero, giving biased estimates. 

clc
clear
global nco cdindex cdid nbund pdid obs id nest

% clear all
% close all
% close hidden
% warning off all

%**************************************************************************
%Loading your data here 
%**************************************************************************

load data2022

%**************************************************************************

% creating counters for %%
obs = size(D_mix,1);       % Number of lines/observations 
nbund = max(cdindex)+1;     % Number of options 
nco = size(D_mix,1)/nbund; % Number of choice occasions;


%%
%**************************************************************************
% Data manipulation
%**************************************************************************

cdindex = [nbund:nbund:obs]';   % observation # of each choice occ.
cdid = repmat(1:nco,[nbund 1]); %
cdid = cdid(:);                 % id for each choice occ.

%The sigma varies with time, such that /sigma_{it} where t are choice 
%occasions. 

%-- Create your product matrix from --%
pdid = [  pdid11 pdid21 pdid31 pdid12 pdid22 pdid32 pdid13 pdid23 pdid33  ];

%**************************************************************************
%Matrix of variables
%**************************************************************************
   
%Product characteristics coefficients
X1 = [ pdid Expenditure ];

% choices: 10, choice occasion: 8192
% no. of ppl: 820
% no. of times: 10 (last one=2)

%%
%**************************************************************************
%%%  Multinomial logit demand - MLE %%%
%**************************************************************************


%%%%%%%%%%%%%%% Optimization algorithm %%%%%%%%%%%%%%%%%%%%%

%Taking random starting values 
XO = zeros(10,1)  ; 

%%%%%%%%%%% Option setting %%%%%%%%
options = optimoptions('fminunc',...
    'TolFun',1e-13,...
    'MaxFunEvals',1e12,...
    'MaxIter',1e12,...
    'TolX',1e-13,...
    'GradObj','off',...        
    'CheckGradients',false,...    
    'FiniteDifferenceType','central',...
    'Display','iter');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
objfunc =@(x) mlogit(x,D_mix,X1);
[theta_mlogit,fval,exitFlag,output] = fminunc(objfunc,XO,options);
fprintf('The number of iterations was : %d\n', output.iterations);
fprintf('The best function value found was : %g\n', fval);
toc

%%
XO = ones(11,1)  ; 

%%%%%%%%%%% Option setting %%%%%%%%
options = optimoptions('fminunc',...
    'TolFun',1e-13,...
    'MaxFunEvals',1e12,...
    'MaxIter',1e12,...
    'TolX',1e-13,...
    'GradObj','off',...        
    'CheckGradients',false,...    
    'FiniteDifferenceType','central',...
    'Display','iter');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
objfunc =@(x) mlogit_true(x,D_mix,X1);
[theta_mlogit,fval,exitFlag,output] = fminunc(objfunc,XO,options);
fprintf('The number of iterations was : %d\n', output.iterations);
fprintf('The best function value found was : %g\n', fval);
toc
%%%%%%%%%%%%%%% END Optimization  %%%%%%%%%%%%%%%%%%%%%


%%

%**************************************************************************
%%%  Multinomial logit demand - NFP %%%
%**************************************************************************

% Nested Fixed point is an algorithm that is helpful when we cannot invert
% choice equation due to non-linearity arising from price endogenity.

X1 = [ pdid Expenditure ];

%%%%%%%%%%%%%%% Optimization algorithm %%%%%%%%%%%%%%%%%%%%%

%Taking random starting values 
XO = rand(size(X1,2)+1 ,1)  ; 
%%%%%%%%%%% Option setting %%%%%%%%
options = optimoptions('fminunc',...
    'TolFun',1e-13,...
    'MaxFunEvals',1e12,...
    'MaxIter',1e12,...
    'TolX',1e-13,...
    'GradObj','off',...        
    'CheckGradients',false,...    
    'FiniteDifferenceType','central',...
    'Display','iter');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
objfunc =@(x) mlogit_fe(x,D_mix,X1);
[theta_fe,fval,exitFlag,output] = fminunc(objfunc,XO,options);
fprintf('The number of iterations was : %d\n', output.iterations);
fprintf('The best function value found was : %g\n', fval);
toc
%%%%%%%%%%%%%%% END Optimization  %%%%%%%%%%%%%%%%%%%%%

[theta_mlogit(end), theta_fe]

