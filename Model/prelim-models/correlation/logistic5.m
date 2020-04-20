function yhat = logistic5(beta,X)
%LOGISTIC - A "function" function to be used with nlinfit in non-linear
%	regression.
%
%	INPUT	X - n x p matrix of independent variable values
%		Beta-	vector of initial estimates of parameters
%
%	OUTPUT	yhat = predicted values for response variable
%The function used is
%
% y(x) = (ymax - ymin)/(1 + e^(-((x - xmean)/Abs[beta]))) + ymin
%
%with the four free (estimated) paramaters: ymin, ymax, xmean, and beta.
%The function is fit to the data, minimizing (y-y(x))^2,
%with:
%        x=      model predictions,
%        y(x) =  transformed model predictions,
%        y=      DMOS.
%
%---------------------------------------------------------------------------
%
a = beta(1); % ymax
b = beta(2); % ymin
c = beta(3); % xmean
s = beta(4);
yhat = (a-b)./(1+ exp(-((X-c)/abs(s)))) + b;
return;
