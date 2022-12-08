% ************************************************************* %
% File name: Demand estimation
% Author(s): Daniel Herrera-Araujo
% Date: 2022
% Description: Estimation of MNL logit using MLE and NFP 
% * * *
% Inputs: datasets on demand
% ************************************************************* %

clc
clear
global nco cdindex cdid nbund pdid obs id nest

% clear all
% close all
% close hidden
% warning off all

%**************************************************************************
%Loading the data
%**************************************************************************

load data2022

%**************************************************************************

% creating vars %%
obs=size(D_mix,1);       % Number of lines/observations 
nbund=max(cdindex)+1;     % Number of options 
nco=size(D_mix,1)/nbund; % Number of choice occasions;


%%
%**************************************************************************
% Data manipulation
%**************************************************************************

cdindex = [nbund:nbund:obs]';   %observation # of each choice occ.
cdid = repmat(1:nco,[nbund 1]); %
cdid = cdid(:);                 %id for each choice occ.

%The sigma varies with time, such that /sigma_{it} where t are choice 
%occasions. 

%-- Product matrix --%
pdid = [  pdid11 pdid21 pdid31 pdid12 pdid22 pdid32 pdid13 pdid23 pdid33  ];

%**************************************************************************
%Matrix of variables
%**************************************************************************
   
%Product characteristics coefficients
X1 = [ pdid Expenditure ];


%%
%**************************************************************************
%%%  Multinomial logit demand - MLE %%%
%**************************************************************************

% True coefficients
theta_true = [3; 2; 1; 1; 0.5; 2; 1; 1; 0.5];


%%%%%%%%%%%%%%% Optimization algorithm %%%%%%%%%%%%%%%%%%%%%

%Taking random starting values 
XO = rand(size(X1,2) ,1)  ; 

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
%%%%%%%%%%%%%%% END Optimization  %%%%%%%%%%%%%%%%%%%%%


%%

%**************************************************************************
%%%  Multinomial logit demand - NFP %%%
%**************************************************************************

X1 = [ Expenditure ];

%%%%%%%%%%%%%%% Optimization algorithm %%%%%%%%%%%%%%%%%%%%%

%Taking random starting values 
XO = rand(size(X1,2) ,1)  ; 

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
objfunc =@(x) mlogit_fe(x,D_mix,X1,pdid);
[theta_fe,fval,exitFlag,output] = fminunc(objfunc,XO,options);
fprintf('The number of iterations was : %d\n', output.iterations);
fprintf('The best function value found was : %g\n', fval);
toc
%%%%%%%%%%%%%%% END Optimization  %%%%%%%%%%%%%%%%%%%%%

[theta_mlogit(end), theta_fe]


