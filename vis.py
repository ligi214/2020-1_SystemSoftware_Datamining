import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./../results/accuracy.csv')
iteration = df.values[:, 0]
accuracy = df.values[:, 1]
gen_accuracy = df.values[:, 2]
distance = df.values[:, 3]

plt.figure()
plt.plot(iteration, accuracy, '-o')
plt.savefig('./../results/accuracy.png')

plt.figure()
plt.plot(iteration, gen_accuracy, '-o')
plt.savefig('./../results/gen_accuracy.png')

plt.figure()
plt.plot(iteration, distance, '-o')
plt.savefig('./../results/distance.png')
plt.show()

df = pd.read_csv('./../results/method.csv')
print(df)

