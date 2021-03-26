import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [20, 14]
plt.rcParams.update({'font.size': 20})

dx = 1 / 1000
x = np.arange(0, 1, dx)
fclean = np.sin(2 * np.pi * 30 * x) + np.sin(2 * np.pi * 90 * x)
f = fclean + np.random.randn(len(x))

n = len(x)
fhat = np.fft.fft(f, n)
PSD = fhat * np.conj(fhat) / n
freq = 1 / (dx * n) * np.arange(n)
L = np.arange(1, np.floor(n/2), dtype=int)

indices = PSD > 100
PSDclean = PSD * indices
fhat = fhat * indices
ffilt = np.fft.ifft(fhat)

fig, axs = plt.subplots(3)

plt.sca(axs[0])
plt.plot(x, fclean, c='black', label='Clean')
plt.plot(x, f, label='Noisy')
plt.xlim(x[0], x[-1])
plt.legend(loc='upper right')

plt.sca(axs[1])
plt.plot(freq[L], PSD[L], c='red')
plt.plot(freq[L], PSDclean[L], c='black', label='Filtered')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend(loc='upper right')

plt.sca(axs[2])
plt.plot(x, ffilt, c='black', label='Clean')
plt.xlim(x[0], x[-1])
plt.legend(loc='upper right')

plt.show()