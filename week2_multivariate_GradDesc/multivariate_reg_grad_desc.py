# WEEKLY FOCUS: Multivariate Regression
# Objectives:
    # 1) First, modify gradient descent to fit a second parameter the intercept, b_0, in b_1*x + b_0
    # 2) Second, Vectorize gradient descent to take any number of parameteters to fit. This fit's a so-called "hyperplane", which is an n-dimensional plane.
    # 3) Plot loss and compare custom gradient descent, to multivariate regression model from sci-kit learn.


def mygradesc(x, y, learning_rate, iterations):
	b = 5 #(1) initialize beta
	n = len(x) #sets n equal to how many data points are in set
	assert len(x) == len(y) #ensure that x and y share the same length
	loss_history = [] #creates a new list
	for i in range(iterations): 
	# calculate the loss for a value of b
		loss_sum = 0
		for j in range(n):
			loss_sum+= (b*x[j]-y[j])**2
		total_loss=(1/n)*loss_sum
	#calculate the derivative of the loss function
		deriv_loss = 0
		for k in range(n):
			deriv_loss+=(b*x[j]-y[j])
		total_deriv = (1/n)*(2*b)*deriv_loss
		b = b - learning_rate*total_deriv
		loss_history.append(total_loss) #adds the current value of 
# total loss to the list, continues until iterations have reached

	return b, loss_history

