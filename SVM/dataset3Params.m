function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
steps = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];

C_best = 1;
sigma_best = 0.3;

model = svmTrain(X, y, C_best, @(x1, x2) gaussianKernel(x1, x2, sigma_best));
predictions = svmPredict(model, Xval);
error_best = mean(double(predictions ~= yval));


for C_for = steps

	for sigma_for = steps
	
		if (C_for ~= 1 && sigma_for ~= 0.3)
	
			model = svmTrain(X, y, C_for, @(x1, x2) gaussianKernel(x1, x2, sigma_for));
			predictions = svmPredict(model, Xval);
			error = mean(double(predictions ~= yval));
	
			if (error < error_best)
				error_best = error
				C_best = C_for
				sigma_best = sigma_for
			endif
	
		endif
	endfor

endfor

C = C_best;
sigma = sigma_best








% =========================================================================

end
