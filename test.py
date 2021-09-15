import AdapTE as apte
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


x = np.random.rand(1000, 200)
df = pd.DataFrame(x)
# print(x)
# print(df)

pte, surr = apte.AdapTE(df, taus=[1, 2], dimEmbs=[1, 2], which=[1], mode='driving', surr='ts')

plt.imshow(pte)
plt.colorbar()
plt.show()

A = np.zeros(np.shape(pte))
for i in range(np.shape(pte)[0]):
    for j in range(np.shape(pte)[1]):
        if pte[i, j] > surr[i, j]:
            A[i, j] = pte[i, j]

plt.imshow(A)
plt.colorbar()
plt.show()
