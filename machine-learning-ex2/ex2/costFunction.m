function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
h = sigmoid(X *theta );
J = 1/m*(-y.*log(h)-(1-y).*log(1-h));
grad =1/m * X' *(h-y);


% num_iters = 100
% j_size = size(theta,1)
% for iter = 1:num_iters
%   for j_num = 1:j_size
%     theta(j_num) = theta(j_num) -alpha * 1/m * (h-y) * X(:,j_num)
%   end
% end







% =============================================================

end
