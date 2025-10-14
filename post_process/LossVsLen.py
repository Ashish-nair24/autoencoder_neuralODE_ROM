# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 22:01:57 2023

@author: ashis
"""

import matplotlib.pyplot as plt


trainLoss = [1.259789228,
1.2882092,
1.295847535,
0.434209913,
0.080949649,
0.004290519,
0.002021446,
0.000859945,
0.00012835,
0.000563524
]

trainLoss2 = [1.94652009,
1.879723668,
1.872224212,
2.328816414,
2.697141886,
0.043222893,
0.111868285,
0.080959469,
0.000829183,
0.000795615
]


trainLoss3 = [2.074106216,
2.008792639,
2.009847164,
5.471935749,
13.77980804,
0.395877421,
1.753222585,
0.460990399,
0.005904194,
0.001783398
]

trainLoss4 = [2.423549652,
2.095635653,
2.052880764,
37.99380493,
20.20614624,
2.315789938,
6.802453041,
1.489873052,
0.025818948,
0.001922301
]

trainLoss5 = [0.533525288,
0.545775712,
0.548762321,
0.047967836,
0.001836854,
0.000101112,
0.000235873,
5.18e-05,
8.82e-05,
0.000603096
]

# testLoss = [2.102958441,
# 2.103862286,
# 2.106814861,
# 5.54483366,
# 2.985970974,
# 2.922276735,
# 4.624382019,
# 3.263430834,
# 2.800266981,
# 2.428338051
# ]

trajLen = [1,
7,
9,
19,
49,
99,
199,
499,
999,
1999,
]

plt.loglog(trajLen, trainLoss5, label='Rlt Lng = 500')
plt.loglog(trajLen, trainLoss,  label='Rlt Lng = 1000')
plt.loglog(trajLen, trainLoss2, label='Rlt Lng = 2000')
plt.loglog(trajLen, trainLoss3, label='Rlt Lng = 3000')
plt.loglog(trajLen, trainLoss4, label='Rlt Lng = 4000')
plt.xlabel('$n_t$')
plt.ylabel('Rollout Loss')
plt.title('Training Samples')
plt.legend()
plt.savefig('TrainLossVsLen.png')
plt.close()

# plt.loglog(trajLen, testLoss)
# plt.xlabel('$n_t$')
# plt.ylabel('Rollout Loss')
# plt.title('Testing Samples')
# plt.savefig('TestLossVsLen.png')
# plt.close()