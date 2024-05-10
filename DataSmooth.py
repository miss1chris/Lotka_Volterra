import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'res_data_log.csv')

# 简单指数平滑
def exponential_smoothing(series, alpha):
  result = [series[0]]
  for n in range(1, len(series)):
    # Y`t+1 = a*Yt + (1-t)*Y`t
    result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
  return result


def plotExponentialSmoothing(series, alphas):
  with plt.style.context('seaborn-white'):
    plt.figure(figsize=(15, 7))
    for alpha in alphas:
      plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
    plt.plot(series.values, color="black", label="Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Exponential Smoothing")
    plt.grid(True)
#         plt.show()


plt.figure(figsize=(10,8))

plt.subplot(1, 2, 1)
plt.plot(exponential_smoothing(df['prey'], 0.5),label="Alpha 0.5",linestyle='-')
plt.plot(df['prey'].values,"c",label = "prey",linestyle='--')


plt.legend(loc="best")
plt.axis('tight')
plt.title("Exponential Smoothing")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(exponential_smoothing(df['predator'], 0.5),label="Alpha 0.5",linestyle='-')
plt.plot(df['predator'].values,color = "c",label = "predator",linestyle='--')
plt.legend(loc="best")
plt.axis('tight')
plt.title("Exponential Smoothing")
plt.grid(True)

plt.suptitle("RUNOOB subplot")
plt.show()


#移动平滑
window = 3
# trail-rolling average transform
rolling = df['prey'].rolling(window=window)
rolling_mean = rolling.mean()
plt.figure(figsize=(15, 7))
plt.plot(rolling_mean,'r',linestyle='-')
plt.plot(df['prey'].values,"c",label = "prey",linestyle='--')
plt.show()