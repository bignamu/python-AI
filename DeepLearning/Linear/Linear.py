import numpy as np
import matplotlib.pyplot as plt

data = [[2,81],[4,93],[6,91],[8,97]]
x = [i[0] for i in data]
y = [i[0] for i in data]

plt.figure(figsize=(8,5))
plt.scatter(x,y)
plt.show()



x_data = np.array(x)
y_data = np.array(y)

a = 0
b = 0

lr = 0.03
epochs = 2001

for i in range(epochs):
    y_pred = a * x_data + b
    error = y_pred - y_data
    # cost = 2/m(시그마)(예측값 - 실제값)**2 미분
    a_diff = (2/len(x_data)) * sum(x_data*(error))
    b_diff = (2/len(x_data)) * sum(error)

    a = a - lr * a_diff
    b = b - lr * b_diff

    if i % 100 == 0:
        print(f'epoch = {i}, diff = {a}, bias = {b}')


y_pred = a * x_data + b
plt.scatter(x,y)
plt.plot([min(x_data),max(x_data)],[min(y_pred),max(y_pred)])
plt.show()