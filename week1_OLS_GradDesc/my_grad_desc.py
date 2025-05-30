def mygradesc(x, y, learning_rate, iterations):
    b1 = 5  # (1a) initialize beta
    b0 = 5 #(1b) initialize the slope intercept
    n = len(x)  # sets n equal to how many data points are in set
    assert len(x) == len(y)  # ensure that x and y share the same length
    loss_history = []  # creates a new list

    for i in range(iterations): 
        # calculate the loss for a value of b
        loss_sum = 0
        deriv_b1 = 0
        deriv_b0 = 0
        for j in range(n):
            obs_error = (b0*1 + b1 * x[j]) - y[j]
            loss_sum += obs_error ** 2
            deriv_b1 += obs_error*x[j]
            deriv_b0 += obs_error
        total_loss = (1 / n) * loss_sum

        # calculate the derivative of the loss function
        total_deriv_b1 = (1 / n)*2* deriv_b1
        total_deriv_b0 = (1 / n)*2* deriv_b0
        b1 = b1 - learning_rate * total_deriv_b1
        b0 = b0 - learning_rate * total_deriv_b0
        loss_history.append(total_loss)  # adds the current value of total loss to the list

    return b0, b1, loss_history
