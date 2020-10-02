import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./../results/TDH+EAI.csv')
iteration = df.values[:, 0]
accuracy_eai = df.values[:, 1]
gen_accuracy_eai = df.values[:, 2]
distance_eai = df.values[:, 3]

df = pd.read_csv('./../results/TDH+RAND.csv')
accuracy_rand = df.values[:, 1]
gen_accuracy_rand = df.values[:, 2]
distance_rand = df.values[:, 3]

plt.figure()
plt.plot(iteration, accuracy_eai, '-o', label='TDH+EAI')
plt.plot(iteration, accuracy_rand, '-x', label='TDH+RAND')
plt.legend()
plt.savefig('./../results/accuracy.png')

plt.figure()
plt.plot(iteration, gen_accuracy_eai, '-o', label='TDH+EAI')
plt.plot(iteration, gen_accuracy_rand, '-x', label='TDH+RAND')
plt.legend()
plt.savefig('./../results/gen_accuracy.png')

plt.figure()
plt.plot(iteration, distance_eai, '-o', label='TDH+EAI')
plt.plot(iteration, distance_rand, '-x', label='TDH+RAND')
plt.legend()
plt.savefig('./../results/distance.png')
plt.show()

df = pd.read_csv('./../results/method.csv')
print(df)

