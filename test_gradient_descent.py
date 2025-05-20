from housing_data import square_footage, housing_prices
import matplotlib.pyplot as plt
import numpy as np

# Normalize Values
X = [l/max(square_footage) for l in square_footage]
y = [l/max(housing_prices) for l in housing_prices]

lr = 0.01
iterations = 1000

b_1, loss_history = my_gradient_descent(X,y,lr,iterations)

# Expect near 1
print(f'Slope fitted {b_1}')

# Plotting data and fitted line

# a. Plotting scatter plot of data
fig,axes= plt.subplots(2,1)
axes[0].scatter(X,y)
axes[0].set_xlabel('Square Footage')
axes[0].set_ylabel('Housing Price')

#b. Plotting fitted line
x_line = np.linspace(0,max(X),100)
y_line = x_line*b_1
axes[0].plot(x_line,y_line,'r')


axes[1].plot(range(iterations),loss_history)
axes[1].set_xlabel('Number of Iterations')
axes[1].set_ylabel('Loss')

plt.show()

