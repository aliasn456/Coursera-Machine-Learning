function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X = [ones(m,1) X];

for i = 1:m,
 	a2(:,i) = sigmoid(Theta1* X(i,:)');
 end

 a2 = [ones(m,1) a2'];

 for j = 1:m,
 	h(:,j) = sigmoid(Theta2*a2(j,:)');
 end



yK = zeros(num_labels, m);


for k = 1:m
    yK(y(k), k) = 1; 
end


Theta1NB = Theta1(:, 2:end);
Theta2NB = Theta2(:, 2:end);


J = sum(sum(-yK .* log(h) - (1-yK) .* log(1-h)))/m...
  + lambda/2/m * (Theta1NB(:)' * Theta1NB(:) + Theta2NB(:)' * Theta2NB(:));


for t = 1:m

	% l=1
	a_1 = X(t,:)';

	% l=2
	z_2 = Theta1 * a_1;
	a_2 = [1; sigmoid(z_2)];

	% l=3
	z_3 = Theta2 * a_2;
	a_3 = sigmoid(z_3);


	yA = ([1:num_labels]==y(t))';
	
	d_3 = a_3 - yA;

	% backing to l=2
	d_2 = (Theta2' * d_3) .* [1; sigmoidGradient(z_2)];
	d_2 = d_2(2 : end); 

	% backing to l=1
	Theta1_grad = Theta1_grad + d_2 * a_1';
	Theta2_grad = Theta2_grad + d_3 * a_2';
end

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end