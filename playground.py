import numpy as  np

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = np.array([1, 2])
# %%
a.shape
# %%
a
# %%
c = a * b
# %%
c[:, :, 1]

