import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./../results/accuracy.csv')
iteration = df.values[:, 0]
accuracy = df.values[:, 1]
gen_accuracy = df.values[:, 2]
distance = df.values[:, 3]

plt.figure()
plt.scatter(iteration, accuracy)

plt.figure()
plt.scatter(iteration, gen_accuracy)

plt.figure()
plt.scatter(iteration, distance)
plt.show()

df = pd.read_csv('./../results/method.csv')
print(df)

