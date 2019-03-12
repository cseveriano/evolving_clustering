import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evolving.datagen import MixtureModelGenerator

# 1. Instantiate the stream generator
generator = MixtureModelGenerator.MixtureModelGenerator(2, 4)
generator.prepare_for_use()

# 2. Get data from the stream
X1, y1 = generator.next_sample(num_samples=100, model_index=0)
X2, y2 = generator.next_sample(num_samples=150, model_index=1)
X3, y3 = generator.next_sample(num_samples=250, model_index=2)
X4, y4 = generator.next_sample(num_samples=100, model_index=3)

X = np.concatenate((X1,X2,X3,X4))
y = np.concatenate((y1,y2,y3,y4))

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=1, cmap='viridis')
plt.show()

# y = y[:,0]
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, s=1, cmap='viridis')
# plt.show()


df = pd.DataFrame(np.hstack((X,y)))
