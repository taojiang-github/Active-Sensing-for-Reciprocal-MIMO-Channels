import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

Na, Nb, Ns, SNR = 64, 64, 4, 0
sigma2 = 10**(-SNR/10)
num_iter = 8
Lp = 10
N_RF = 8

PATH_results = '../Results/rnn_na%d_nb%d_ns%d_iter%d_%ddB_Lp%d_Nrf%d' \
             %(Na, Nb, Ns, num_iter, SNR, Lp, N_RF)+'.mat'
data = sio.loadmat(PATH_results)
rnn = data['rate_rnn'][0]
opt = data['rate_opt'][0, 0]

# PATH_results = '../Results/power_method_na%d_nb%d_ns%d_iter%d_' \
#                '%ddB_Lp%d' %(Na, Nb, Ns, num_iter, SNR, Lp)+'.mat'
# data = sio.loadmat(PATH_results)
# pwoer_methods = data['rate_power'][0]
#
PATH_results = '../Results/codebook_power_method_na%d_nb%d_ns%d_iter%d_' \
               '%ddB_Lp%d_NRF%d' % (Na, Nb, Ns, num_iter, SNR, Lp, N_RF) + '.mat'
data = sio.loadmat(PATH_results)
codebook = data['rate_power'][0]

omp = []
num_grids = 64

##for Ns=4
num_iter_omp = np.array([1,2,4,6,8])
len1 = [2,4,4,6,8]
len2 = [2,4,8,8,8]
grid = True
for ii in range(len(num_iter_omp)):
    if grid:
        PATH_results = '../Results/omp_grid_na%d_nb%d_ns%d_iter%d_' \
                       '%ddB_Lp%d_grids%d_NRF_%d_pilot_%d_%d' % (
                           Na, Nb, Ns, num_iter_omp[ii], SNR, Lp, num_grids, N_RF,
                           len1[ii], len2[ii]) + '.mat'
    else:
        PATH_results = '../Results/omp_gridless_na%d_nb%d_ns%d_iter%d_' \
                       '%ddB_Lp%d_grids%d_NRF_%d_pilot_%d_%d' % (
                           Na, Nb, Ns, num_iter_omp[ii], SNR, Lp, num_grids, N_RF,
                           len1[ii], len2[ii]) + '.mat'
    data = sio.loadmat(PATH_results)
    omp.append(data['rate_omp'][0][0])

num_iter = 8
PATH_results = '../Results/power_method_MTQR_na%d_nb%d_ns%d_iter%d_%ddB' \
             %(Na, Nb, Ns, num_iter, SNR)
data = sio.loadmat(PATH_results)
mtqr = data['rate_power'][0]

plt.plot(np.arange(1,9)*Ns*2,opt*np.ones(8),label='Optimal')
plt.plot(np.arange(1,9)*Ns*2,rnn,'s-', label='Proposed active sensing',markersize=8)
plt.plot(np.arange(1,9)*Ns*2, codebook,'o-', label='Power iteration with codebook',markersize=8)
plt.plot(np.arange(1,2)*Ns*2*8, mtqr[0:1],'>-', label='Modified Two-way QR (MTQR)',markersize=8)
plt.plot(num_iter_omp*Ns*2, omp,'v-', label='OMP+SVD',markersize=8)
plt.xticks(np.arange(2,65,4))
plt.grid()
plt.legend(fontsize=12)
plt.xlabel('Pilot length $2LN_{\\rm s}$', fontsize=14)
plt.ylabel('Average objective value '+ r'$\log\det(\bm F_{\rm r}^{\sf H}\hat{\bm W}_{\rm r}^{\sf H} \bm G\bm F_{\rm t}\hat{\bm W}_{\rm t})$', fontsize=14)
plt.savefig('SNR%ddB_Ns%d'%(SNR,Ns)+'.pdf',bbox_inches='tight')
plt.show()