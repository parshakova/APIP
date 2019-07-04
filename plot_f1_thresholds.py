import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from collections import OrderedDict

sns.set()

m_10 = {1:OrderedDict([(0.1, 91.79), (0.2, 91.37), (0.3, 90.58), (0.4, 89.71), (0.5, 87.79), (0.6, 87.28), (0.7, 84.81), (0.8, 83.44), (0.9, 82.87)]),\
		2:OrderedDict([(0.1, 76.32), (0.2, 74.01), (0.3, 70.74), (0.4, 64.73), (0.5, 56.96), (0.6, 53.9), (0.7, 36.37), (0.8, 28.16), (0.9, 23.36)]) ,\
		3:OrderedDict([(0.1, 63.6), (0.2, 59.85), (0.3, 54.96), (0.4, 46.76), (0.5, 38.12), (0.6, 34.18), (0.7, 20.84), (0.8, 12.96), (0.9, 7.24)]) }

m_8 = { 1:OrderedDict([(0.1, 92.9), (0.2, 92.38), (0.3, 91.67), (0.4, 90.8), (0.5, 89.12), (0.6, 88.53), (0.7, 86.39), (0.8, 85.22), (0.9, 84.71)]),\
		2:OrderedDict([(0.1, 78.68), (0.2, 76.43), (0.3, 73.49), (0.4, 67.7), (0.5, 59.96), (0.6, 57.08), (0.7, 39.84), (0.8, 31.8), (0.9, 26.81)]) ,\
		3:OrderedDict([(0.1, 66.14), (0.2, 62.52), (0.3, 58.39), (0.4, 50.19), (0.5, 40.47), (0.6, 36.53), (0.7, 23.7), (0.8, 15.06), (0.9, 9.09)]) }

m_5 = { 1:OrderedDict([(0.1, 91.41), (0.2, 91.0), (0.3, 90.25), (0.4, 89.12), (0.5, 87.26), (0.6, 86.69), (0.7, 84.04), (0.8, 82.73), (0.9, 82.18)]),\
		2:OrderedDict([(0.1, 74.72), (0.2, 72.55), (0.3, 69.26), (0.4, 63.15), (0.5, 55.27), (0.6, 52.44), (0.7, 34.15), (0.8, 25.87), (0.9, 20.94)]),\
		3:OrderedDict([(0.1, 60.99), (0.2, 57.05), (0.3, 52.16), (0.4, 44.22), (0.5, 34.88), (0.6, 30.75), (0.7, 17.98), (0.8, 10.36), (0.9, 5.02)])}

m_4 = { 1: OrderedDict([(0.1, 91.34), (0.2, 90.88), (0.3, 90.27), (0.4, 89.26), (0.5, 87.2), (0.6, 86.65), (0.7, 84.21), (0.8, 82.79), (0.9, 82.18)]),\
		2:OrderedDict([(0.1, 75.72), (0.2, 73.23), (0.3, 70.13), (0.4, 63.88), (0.5, 55.98), (0.6, 52.92), (0.7, 35.02), (0.8, 26.95), (0.9, 22.06)]),\
		3:OrderedDict([(0.1, 61.69), (0.2, 57.75), (0.3, 53.3), (0.4, 45.49), (0.5, 36.28), (0.6, 31.89), (0.7, 19.12), (0.8, 10.99), (0.9, 5.84)]) }

m_3 = { 1: OrderedDict([(0.1, 89.54), (0.2, 88.98), (0.3, 88.03), (0.4, 87.06), (0.5, 85.2), (0.6, 84.61), (0.7, 81.74), (0.8, 80.26), (0.9, 79.78)]),\
		2:OrderedDict([(0.1, 73.19), (0.2, 70.56), (0.3, 67.25), (0.4, 60.89), (0.5, 52.73), (0.6, 49.72), (0.7, 31.31), (0.8, 23.4), (0.9, 18.84)]),\
		3:OrderedDict([(0.1, 59.02), (0.2, 54.96), (0.3, 50.19), (0.4, 40.91), (0.5, 31.89), (0.6, 27.83), (0.7, 15.69), (0.8, 9.02), (0.9, 4.57)])}

drqa = {1:OrderedDict([(0.1, 87.34), (0.2, 86.88), (0.3, 85.97), (0.4, 84.95), (0.5, 83.01), (0.6, 82.35), (0.7, 79.25), (0.8, 77.63), (0.9, 76.97)]),\
		2:OrderedDict([(0.1, 69.05), (0.2, 66.29), (0.3, 62.88), (0.4, 56.57), (0.5, 48.21), (0.6, 45.24), (0.7, 27.11), (0.8, 19.77), (0.9, 15.56)]),\
		3:OrderedDict([(0.1, 54.45), (0.2, 50.06), (0.3, 45.24), (0.4, 35.96), (0.5, 27.26), (0.6, 23.25), (0.7, 12.2), (0.8, 6.35), (0.9, 2.73)])}



rnet = {1:{0.1: 86.13, 0.2: 85.68, 0.3: 84.85, 0.4: 84.02, 0.5: 81.9, 0.6: 81.29, 0.7: 78.26, 0.8: 76.91, 0.9: 76.24}, \
		2:{0.1: 68.66, 0.2: 66.13, 0.3: 63.29, 0.4: 56.53, 0.5: 48.2, 0.6: 45.42, 0.7: 26.92, 0.8: 19.2, 0.9: 14.97},\
		3:{0.1: 53.49, 0.2: 48.98, 0.3: 44.6, 0.4: 35.96, 0.5: 27.51, 0.6: 24.02, 0.7: 12.39, 0.8: 5.84, 0.9: 2.41}}

bidafe = {1:{0.1: 87.2, 0.2: 86.86, 0.3: 86.29, 0.4: 85.06, 0.5: 83.13, 0.6: 82.53, 0.7: 79.9, 0.8: 78.48, 0.9: 77.9},\
		  2:{0.1: 69.88, 0.2: 67.64, 0.3: 64.73, 0.4: 57.99, 0.5: 49.26, 0.6: 46.22, 0.7: 27.75, 0.8: 20.32, 0.9: 15.88}, \
		  3:{0.1: 53.49, 0.2: 50.13, 0.3: 45.62, 0.4: 37.29, 0.5: 29.1, 0.6: 24.65, 0.7: 12.96, 0.8: 6.48, 0.9: 2.67}}

qanet = {1:{0.1: 87.66, 0.2: 86.92, 0.3: 86.09, 0.4: 84.59, 0.5: 82.47, 0.6: 81.84, 0.7: 78.81, 0.8: 76.91, 0.9: 76.32}, \
		 2:{0.1: 70.13, 0.2: 67.45, 0.3: 63.93, 0.4: 56.82, 0.5: 48.34, 0.6: 45.39, 0.7: 27.5, 0.8: 19.63, 0.9: 15.15}, \
		 3:{0.1: 54.96, 0.2: 50.95, 0.3: 45.55, 0.4: 36.66, 0.5: 28.02, 0.6: 24.33, 0.7: 12.52, 0.8: 6.29, 0.9: 2.73}}

mnem = {1:{0.1: 88.68, 0.2: 88.29, 0.3: 87.62, 0.4: 86.53, 0.5: 84.57, 0.6: 84.15, 0.7: 81.11, 0.8: 79.62, 0.9: 78.95}, \
		2:{0.1: 70.31, 0.2: 68.02, 0.3: 65.08, 0.4: 58.4, 0.5: 49.49, 0.6: 46.67, 0.7: 27.95, 0.8: 20.2, 0.9: 16.05},\
		3:{0.1: 53.68, 0.2: 50.51, 0.3: 45.87, 0.4: 37.48, 0.5: 28.72, 0.6: 24.27, 0.7: 12.58, 0.8: 6.04, 0.9: 2.86}}

for d in [rnet, bidafe, qanet, mnem]:
	for i in range(3):
		dummy = OrderedDict(sorted(d[i+1].items(), key=lambda t: t[0]))
		d[i+1] = dummy

df = pd.DataFrame({'th':zip(*m_10[1].items())[0], 'i10_a1':np.array(zip(*m_10[1].items())[1]), \
					'i10_a2':np.array(zip(*m_10[2].items())[1]), 'i10_a3':np.array(zip(*m_10[3].items())[1]),\
					'i8_a1':np.array(zip(*m_8[1].items())[1]), 'i8_a2':np.array(zip(*m_8[2].items())[1]), 'i8_a3':np.array(zip(*m_8[3].items())[1]),\
					'i5_a1':np.array(zip(*m_5[1].items())[1]), 'i5_a2':np.array(zip(*m_5[2].items())[1]), 'i5_a3':np.array(zip(*m_5[3].items())[1]),\
					'i3_a1':np.array(zip(*m_3[1].items())[1]), 'i3_a2':np.array(zip(*m_3[2].items())[1]), 'i3_a3':np.array(zip(*m_3[3].items())[1]),
					'i4_a1':np.array(zip(*m_4[1].items())[1]), 'i4_a2':np.array(zip(*m_4[2].items())[1]), 'i4_a3':np.array(zip(*m_4[3].items())[1]),
					'RNet_a1':np.array(zip(*rnet[1].items())[1]), 'RNet_a2':np.array(zip(*rnet[2].items())[1]), 'RNet_a3':np.array(zip(*rnet[3].items())[1]),
					'DrQA_a1':np.array(zip(*drqa[1].items())[1]), 'DrQA_a2':np.array(zip(*drqa[2].items())[1]), 'DrQA_a3':np.array(zip(*drqa[3].items())[1]),
					'BiDAF_a1':np.array(zip(*bidafe[1].items())[1]), 'BiDAF_a2':np.array(zip(*bidafe[2].items())[1]), 'BiDAF_a3':np.array(zip(*bidafe[3].items())[1]),
					'QANet_a1':np.array(zip(*qanet[1].items())[1]), 'QANet_a2':np.array(zip(*qanet[2].items())[1]), 'QANet_a3':np.array(zip(*qanet[3].items())[1]),
					'Mnem_a1':np.array(zip(*mnem[1].items())[1]), 'Mnem_a2':np.array(zip(*mnem[2].items())[1]), 'Mnem_a3':np.array(zip(*mnem[3].items())[1])})
lw = 2; ms = 0

plt.subplot(1,2, 1)
plt.plot( 'th', 'i10_a1', data=df, marker='^', markersize=ms, color='dodgerblue', linewidth=lw, linestyle='-.')
plt.plot( 'th', 'i8_a1', data=df, marker='^', markersize=ms, color='salmon', linewidth=lw, linestyle='-.')
plt.plot( 'th', 'i5_a1', data=df, marker='^', markersize=ms, color='mediumvioletred', linewidth=lw, linestyle='-.')
plt.plot( 'th', 'i4_a1', data=df, marker='^', markersize=ms, color='mediumseagreen', linewidth=lw, linestyle='-.')
plt.plot( 'th', 'i3_a1', data=df, marker='^', markersize=ms, color='gold', linewidth=lw, linestyle='-.')

plt.plot( 'th', 'i10_a2', data=df, marker='o', markersize=ms, color='dodgerblue', linewidth=lw)
plt.plot( 'th', 'i8_a2', data=df, marker='o', markersize=ms, color='salmon', linewidth=lw)
plt.plot( 'th', 'i5_a2', data=df, marker='o', markersize=ms, color='mediumvioletred', linewidth=lw)
plt.plot( 'th', 'i4_a2', data=df, marker='o', markersize=ms, color='mediumseagreen', linewidth=lw)
plt.plot( 'th', 'i3_a2', data=df, marker='o', markersize=ms, color='gold', linewidth=lw)

plt.plot( 'th', 'i10_a3', data=df, marker='o', markersize=ms, color='dodgerblue', linewidth=lw, linestyle=':')
plt.plot( 'th', 'i8_a3', data=df, marker='o', markersize=ms, color='salmon', linewidth=lw, linestyle=':')
plt.plot( 'th', 'i5_a3', data=df, marker='o', markersize=ms, color='mediumvioletred', linewidth=lw, linestyle=':')
plt.plot( 'th', 'i4_a3', data=df, marker='o', markersize=ms, color='mediumseagreen', linewidth=lw, linestyle=':')
plt.plot( 'th', 'i3_a3', data=df, marker='o', markersize=ms, color='gold', linewidth=lw, linestyle=':')
plt.legend()
plt.ylim(0,100)
plt.xlabel("F1 Threshold")
plt.ylabel("Ratio of valid anwers")
plt.title("Comparison among APSN flavors")

plt.subplot(1,2, 2)
plt.plot( 'th', 'RNet_a2', data=df, marker='^', markersize=ms, color='dodgerblue', linewidth=lw)
plt.plot( 'th', 'DrQA_a2', data=df, marker='^', markersize=ms, color='salmon', linewidth=lw)
plt.plot( 'th', 'i8_a2', data=df, marker='^', markersize=ms, color='black', linewidth=lw) 
plt.plot( 'th', 'BiDAF_a2', data=df, marker='^', markersize=ms, color='mediumvioletred', linewidth=lw)
plt.plot( 'th', 'QANet_a2', data=df, marker='^', markersize=ms, color='mediumseagreen', linewidth=lw)
plt.plot( 'th', 'Mnem_a2', data=df, marker='^', markersize=ms, color='gold', linewidth=lw)

plt.plot( 'th', 'RNet_a1', data=df, marker='^', markersize=ms, color='dodgerblue', linewidth=lw, linestyle='-.')
plt.plot( 'th', 'DrQA_a1', data=df, marker='^', markersize=ms, color='salmon', linewidth=lw, linestyle='-.')
plt.plot( 'th', 'i8_a1', data=df, marker='^', markersize=ms, color='black', linewidth=lw, linestyle='-.')
plt.plot( 'th', 'BiDAF_a1', data=df, marker='^', markersize=ms, color='mediumvioletred', linewidth=lw, linestyle='-.')
plt.plot( 'th', 'QANet_a1', data=df, marker='^', markersize=ms, color='mediumseagreen', linewidth=lw, linestyle='-.')
plt.plot( 'th', 'Mnem_a1', data=df, marker='^', markersize=ms, color='gold', linewidth=lw, linestyle='-.')

plt.plot( 'th', 'RNet_a3', data=df, marker='^', markersize=ms, color='dodgerblue', linewidth=lw, linestyle=':')
plt.plot( 'th', 'DrQA_a3', data=df, marker='^', markersize=ms, color='salmon', linewidth=lw, linestyle=':')
plt.plot( 'th', 'i8_a3', data=df, marker='^', markersize=ms, color='black', linewidth=lw, linestyle=':')
plt.plot( 'th', 'BiDAF_a3', data=df, marker='^', markersize=ms, color='mediumvioletred', linewidth=lw, linestyle=':')
plt.plot( 'th', 'QANet_a3', data=df, marker='^', markersize=ms, color='mediumseagreen', linewidth=lw, linestyle=':')
plt.plot( 'th', 'Mnem_a3', data=df, marker='^', markersize=ms, color='gold', linewidth=lw, linestyle=':')

plt.ylim(0,100)
plt.legend()
plt.xlabel("F1 Threshold")
plt.ylabel("Ratio of valid anwers")
plt.title("Comparison among other models")

plt.show()