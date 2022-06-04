import numpy as np


preds_10 = np.load("preds_true_10.03.2022-133103.npy")
preds_15 = np.load("preds_true_15.03.2022-181233.npy")
preds_28 = np.load("preds_true_28.03.2022-110808.npy")

print(preds_10[:10,:])
print(preds_15[:10,:])
print(preds_28[:10,:])