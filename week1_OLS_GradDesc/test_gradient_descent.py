from housing_data import square_footage, housing_prices
from my_grad_desc import mygradesc
import matplotlib.pyplot as plt
import numpy as np

# Normalize Values
X = [l/max(square_footage) for l in square_footage]
y = [l/max(housing_prices) for l in housing_prices]

# Try different learning rates
# case 1) learning_rate too big. Loss bounces around, see oscillating graph
# case 2) learning_rate too small. Takes too long to converge, poor result or takes many iterations
# case 3) goldie locks just right
lr = 0.1
iterations = 1000

b_1, loss_history = mygradesc(X,y,lr,iterations)

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

