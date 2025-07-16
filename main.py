import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x= np.random.rand(100, 1) * 10
y= 3 * x.flatten() + np.random.randn(100) * 2

x_train, x_text, y_train, y_test= train_test_split(x, y, test_size= 0.2)

model= LinearRegression()
model.fit(x_train, y_train)

print('zaribe model:', model.coef_[0])
print('arz az mabda:', model.intercept_)

plt.scatter(x, y, label= 'data')
plt.plot(x, model.predict(x), color= 'red', label= 'model')
plt.title('line regresion with accidental data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()