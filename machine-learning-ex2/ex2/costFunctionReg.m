function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
for i=1:m
  J+=((-y(i))*log(sigmoid(theta'*transpose(X(i,:))))-(1-y(i))*(log(1-(sigmoid(theta'*transpose(X(i,:)))))));
end
J=J/m;
reg=0;
for i=2:length(theta)
  reg=reg+theta(i)^2;
  end
reg=(lambda*reg)/(2*m);
J=J+reg;
%% calculating gradient

 for j=1:length(theta)
   sum=0;
    for i=1:m
      sum+=(sigmoid(theta'*transpose(X(i,:)))-y(i))*X(i,j);
    end    
    if j!=1
      grad(j)=(sum/m)+(lambda/m)*theta(j);
    else
      grad(j)=(sum/m);
      end
  end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
