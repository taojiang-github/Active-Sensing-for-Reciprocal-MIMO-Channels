import torch
import torch.linalg as LA
import math
import numpy as np
import scipy
from generate_data import generate_channel_grid,generate_dict,generate_channel
from metrics import compute_logdet_Y
import matplotlib.pyplot as plt
import scipy.io as sio
import time
torch.set_grad_enabled(False)

'System parameter'
Na, Nb, Ns, SNR = 64, 64, 4, 10
sigma2 = 10**(-SNR/10)
num_iter = 2
Lp = 10
N_RF = 8

'Algorithm parameter'
use_cuda = False
device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
print(device)
bsz = 1000
num_grids = 64
# pilot_len = Ns*num_iter*2
# pilot_len1 = int(math.sqrt(pilot_len))
# pilot_len2 = pilot_len//pilot_len1
pilot_len1 = 2
pilot_len2 = 8
'Validation data'
grid = True
if grid:
    H, alpha, idx1, idx2 = generate_channel_grid(bsz, Na, Nb, Lp,
                                             num_grids, random_seed=2023)
    PATH_results = './Results/omp_grid_na%d_nb%d_ns%d_iter%d_' \
                   '%ddB_Lp%d_grids%d_NRF_%d_pilot_%d_%d' % (
                   Na, Nb, Ns, num_iter, SNR, Lp, num_grids, N_RF,pilot_len1,pilot_len2) + '.mat'
else:
    H = generate_channel(bsz, Na, Nb, Lp, random_seed=2023).to(device)
    PATH_results = './Results/omp_gridless_na%d_nb%d_ns%d_iter%d_' \
                   '%ddB_Lp%d_grids%d_NRF_%d_pilot_%d_%d' % (
                   Na, Nb, Ns, num_iter, SNR, Lp, num_grids, N_RF,pilot_len1,pilot_len2) + '.mat'


A_dic1, A_dic2 = generate_dict(Na, Nb, num_grids)
A_dic = np.kron(A_dic1.numpy().conj(), A_dic2.numpy())
# idx = idx1+idx2*num_grids

# A_dic2 = func_codedesign(num_grids, Na, Nb,phi_min=-math.pi/3,
#                         phi_max=math.pi/3)
# A_dic_torch = torch.tensor(A_dic)
# tmp0 = A_dic1[:,idx1[0]]@torch.diag_embed(alpha[0,:,0,0])\
#       @A_dic2[:,idx2[0]].mH/math.sqrt(Lp)
# print(torch.norm(tmp0-H[0].mH))
# tmp = A_dic_torch[:, idx[0]] @ torch.diag_embed(alpha[0,:,0,0])
# tmp = tmp.sum(1,keepdims=True) / math.sqrt(Lp)
# vec_h = torch.reshape(H[0].conj(),(-1,1))
# print(torch.norm(tmp-vec_h))

U, S, Vh = torch.linalg.svd(H.to('cpu'), full_matrices=False)
V = Vh.mH
Wb0 = U[:, :, 0:Ns].to(device)
Wa0 = V[:, :, 0:Ns].to(device)
rate_opt = compute_logdet_Y(Wa0, Wb0, H)
print("rate_opt=%.4f" % rate_opt)

'generate sensing beamformers'


FWa = torch.randn(bsz,Na,pilot_len1,device=device)+\
     1j*torch.randn(bsz,Na,pilot_len1,device=device)
FWa = FWa/LA.norm(FWa,dim=1,keepdims=True)
Fb = torch.randn(bsz,Na,N_RF*pilot_len2,device=device)+\
     1j*torch.randn(bsz,Na,N_RF*pilot_len2,device=device)
Fb = Fb/Fb.abs()

# Wb = torch.randn(1,N_RF,Ns*num_iter,device=device)+\
#      1j*torch.randn(1,N_RF,Ns*num_iter,device=device)
# Fb = torch.randn(1,Nb,N_RF,device=device)+\
#      1j*torch.randn(1,Nb,N_RF,device=device)
# Fb = Fb/Fb.abs()
# FWb = Fb@Wb
# Fb_np = Fb[0].numpy().T
# FWb = FWb/LA.norm(FWb,dim=1,keepdims=True)
# FWb_np = FWb[0].numpy().T.conj()

'A transmits to B'
yb_noiseless = H@FWa
noise = torch.randn(yb_noiseless.shape, device=device,
                    dtype=torch.complex64) * math.sqrt(sigma2)
yb = Fb.mH@(yb_noiseless+noise)
yb_vec = torch.reshape(yb,(bsz,-1,1)).numpy().conj()
'OMP'
for ii in range(bsz):
    Fb_np = Fb[ii].numpy().T
    FWa_np = FWa[ii].numpy().T.conj()
    ts = time.time()
    IW = np.kron(Fb_np,FWa_np)
    A = IW@A_dic
    y = yb_vec[ii]*math.sqrt(Lp)

    # tmp = A_dic_torch[:, idx[ii]] @ torch.diag_embed(alpha[ii, :, 0, 0])
    # tmp = tmp.sum(1, keepdims=True) / math.sqrt(Lp)
    # y0 = IW@tmp.numpy()
    # print(np.linalg.norm(y0-y))

    idx_est = np.zeros(Lp, dtype=int)
    idx1_est = np.zeros(Lp, dtype=int)
    idx2_est = np.zeros(Lp, dtype=int)
    r = y
    for tt in range(Lp):
        lamda_t = np.argmax(np.abs(np.transpose(np.conj(r)) @ A))
        idx_est[tt] = lamda_t
        idx1_est[tt], idx2_est[tt] = np.mod(lamda_t, num_grids),lamda_t // num_grids
        phi_A = np.reshape(A[:, idx_est[0:tt + 1]], (-1, tt + 1))
        alpha_hat = np.linalg.inv(
            np.transpose(np.conj(phi_A)) @ phi_A) @ np.transpose(
            np.conj(phi_A)) @ y
        r = y - phi_A @ alpha_hat
    alpha_hat = torch.tensor(alpha_hat[:,0],dtype=torch.complex64)
    # print(np.sort(idx1_est)==idx1[ii].numpy())
    # print(np.sort(idx2_est)==idx2[ii].numpy())
    H_est = A_dic1[:,idx1_est]@torch.diag_embed(alpha_hat)\
            @A_dic2[:,idx2_est].mH/math.sqrt(Lp)
    H_est = H_est.mH
    # print(ii, LA.norm(H_est-H[ii])/LA.norm(H[ii]))
    if ii==0:
        H_est_all = H_est[None,:,:]
    else:
        H_est_all = torch.cat([H_est_all,H_est[None,:,:]],dim=0)

    print('ii=%d,time=%.2f sec'%(ii,time.time()-ts))

    if ii%20==0:
        'SVD for estimated H'
        U, S, Vh = torch.linalg.svd(H_est_all.to('cpu'), full_matrices=False)
        V = Vh.mH
        Wb0 = U[:, :, 0:Ns].to(device)
        Wa0 = V[:, :, 0:Ns].to(device)
        S0 = torch.diag_embed(S[:,0:Ns].to(device))+0j
        rate_omp = compute_logdet_Y(Wa0, Wb0, H[0:ii+1,:,:])
        print("rate_opt=%.4f, rate_omp=%.4f" % (rate_opt,rate_omp))


        sio.savemat(PATH_results,
                    {'rate_opt': rate_opt.item(), 'rate_omp': rate_omp.item(),
                     'bsz': bsz, 'num_iter': num_iter})









# PATH_results = './Results/omp_svd_na%d_nb%d_ns%d_avg_num%d_iter%d_' \
#                '%ddB_Lp%d' %(Na, Nb, Ns, avg_num, num_iter, SNR, Lp)+'.mat'
# sio.savemat(PATH_results, {'rate_opt':rate_opt.item(),'rate_power':rate_all,
#                            'bsz':bsz,'num_iter':num_iter})
