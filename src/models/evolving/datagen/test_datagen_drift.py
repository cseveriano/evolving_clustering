import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evolving.datagen import MixtureModelGeneratorDrift
from collections import Counter

# 1. Instantiate the stream generator


# Experiment A: abrupt concept drift
# 1 - Samplear 1000 instancias aleatorias
# 2 - Inserir drifts na media de magnitudes (0, 0.4, 0.6, 0.8)
# 3 - Samplear 4000 instancias
dimensions = 2
num_models_pre = 4
num_models_post = 4
num_instances_pre_drift = 1000
drift_duration = 0
drift_magnitude = 1.2

generator = MixtureModelGeneratorDrift.MixtureModelGeneratorDrift(dimensions = dimensions,
num_models_pre = num_models_pre,
num_models_post = num_models_post,
num_instances_pre_drift = num_instances_pre_drift,
drift_duration = drift_duration,
drift_magnitude = drift_magnitude)


generator.prepare_for_use()

X1, y1 = generator.next_sample(num_samples=num_instances_pre_drift)
plt.figure()
plt.scatter(X1[:, 0], X1[:, 1], c=y1, s=1, cmap='viridis')
plt.show()

X2, y2 = generator.next_sample(num_samples=num_instances_pre_drift)

X = np.concatenate((X1, X2), axis=0)
y = np.concatenate((y1, y2), axis=0)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=1, cmap='viridis')
plt.show()

# Experiment B: incremental concept drift
# 1 - Samplear 1000 instancias aleatorias
# 2 - Adicionar coeficiente de aumento com duraçao (1000, 9000)
# 3 - Samplear 4000 instancias

# Experiment C: concept evolution
# 1 - Samplear 1000 instancias aleatorias com 4 classes
# 2 - Alterar os modelos para um novo numero de classes com diferentes distribuicoes (2, 6)
# 3 - Samplear 4000 instancias

