clear
clc

%% Exercice #1 %%

% Import dataset : imports-85.cvs using in-house matlab import commands and
% then save it to imports.mat

% Upload the imports.mat
opts=detectImportOptions('brdata.csv');
opts.VariableNamesLine = 1;
orig_data=readtable('brdata.csv',opts, 'ReadVariableNames', true);
% limting the no. of entrants to max 4 by filtering the input file
idx=orig_data.tire<=4;
new_data=orig_data(idx,:);

% no. of simulations is 1000
ns=1000;
% There are 19 parameters-9 market level (population,etc) and 10 firm level
% coefficients [2 for each firm (variable profit + fixed cost eq.) and
% there are 5 cases of entry(0-4 potential entrants)]

x0 = ones(19,1);
options =  optimset('TolFun',1e-12,...
    'MaxFunEvals',1e10,...
    'MaxIter',1e10,...
    'TolX',1e-12,...
    'Display','off') ;
tic
objfunc =@(x) loglik_avishek(x,new_data,ns);
[theta_sim,fval,exitFlag,output] = fminunc(objfunc,x0,options);
fprintf('The number of iterations was : %d\n', output.iterations);
fprintf('The best function value found was : %g\n', fval);
toc