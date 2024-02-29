import scipy.io as sio
import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

import matplotlib.pylab as plt

Na, Nb, Ns, num_iter, SNR = 64, 64, 4, 16, -10
'Plot rate vs num_iter at 0dB'
lmmse_svd = []
lmmse_pilots = np.array([1,4,8,12,16])
for ii in lmmse_pilots:
    PATH_results = '../Results/lmmse_svd_na%d_nb%d_ns%d_iter%d_%ddB' \
                 %(Na, Nb, Ns, ii, SNR)+'.mat'
    data = sio.loadmat(PATH_results)
    lmmse_svd.append(data['rate_lmmse'][0, 0])

'Power method'
PATH_results = '../Results/power_method_na%d_nb%d_ns%d_iter%d_%ddB' % (
    Na, Nb, Ns, num_iter, SNR) + '.mat'
data = sio.loadmat(PATH_results)
power_method=data['rate_power'][0]

'Power_sum method'
PATH_results = '../Results/power_method_sum_na%d_nb%d_ns%d_iter%d_' \
               '%ddB'  %(Na, Nb, Ns, num_iter, SNR)+'.mat'
data = sio.loadmat(PATH_results)
power_sum_method=data['rate_power'][0]

'Active Sensing'
PATH_results = '../Results/rnn_na%d_nb%d_ns%d_iter%d_%ddB' \
             %(Na, Nb, Ns, num_iter, SNR)+'.mat'
data = sio.loadmat(PATH_results)
rnn = data['rate_rnn'][0]
opt = data['rate_opt'][0][0]


pilots_len = np.arange(1,17)*Ns*2
plt.plot(pilots_len,opt*np.ones(16),label='Optimal',markersize=8)
plt.plot(pilots_len,rnn,'s-', label='Proposed active sensing method',markersize=8)
plt.plot(pilots_len[1:],power_sum_method[1:],'>-', label='Summed power method',markersize=8)
plt.plot(pilots_len,power_method,'o-', label='Power iteration method',markersize=8)
plt.plot(lmmse_pilots*Ns*2,lmmse_svd,'v-',label='LMMSE+SVD',markersize=8)

plt.xlabel('Pilot length $2LN_{\\rm s}$', fontsize=14)
plt.ylabel('Average objective value '+ r'$\log\det(\bm W_{\rm r}^{\sf H} \bm G\bm W_{\rm t})$', fontsize=14)
plt.legend(fontsize=12,loc=2)
plt.xticks(np.arange(Ns*2,Ns*16*2+1,8))
plt.grid()
plt.savefig('fully_d_%ddB%dNs.pdf'%(SNR,Ns),bbox_inches='tight')
plt.show()
