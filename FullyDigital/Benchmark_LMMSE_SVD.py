import torch
import torch.linalg as LA
import math
from generate_data import generate_channel
from metrics import compute_logdet_Y, compute_dist
import scipy.io as sio
import matplotlib.pyplot as plt
torch.set_grad_enabled(False)

'System parameter'
Na, Nb, Ns, num_iter, SNR = 64, 64, 2, 16, 5
sigma2 = 10**(-SNR/10)

'Algorithm parameter'
use_cuda = False
device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
print(device)
bsz = 1000
num_bsz = 1

'Validation data'
H_val = generate_channel(bsz*num_bsz, Na, Nb, random_seed=2023).to(device)

rate_opt, rate_lmmse, dist_a, dist_b = 0.0, 0.0, 0.0, 0.0
for ii in range(num_bsz):
    H = H_val[ii*bsz:bsz*(ii+1)]
    'Optimal'
    U, S, Vh = torch.linalg.svd(H.to('cpu'), full_matrices=False)
    V = Vh.mH
    Wb0 = U[:, :, 0:Ns].to(device)
    Wa0 = V[:, :, 0:Ns].to(device)
    rate_opt0 = compute_logdet_Y(Wa0, Wb0, H)
    print("ii=%d, rate_opt=%.4f" %(ii, rate_opt0))
    rate_opt = rate_opt+rate_opt0

    'LMMSE: B transmits to A'
    Wb = torch.randn(bsz,Nb,Ns*num_iter*2,device=device)+\
         1j*torch.randn(bsz,Nb,Ns*num_iter*2,device=device)
    Wb = Wb/LA.norm(Wb,dim=1,keepdims=True)
    ya_noiseless = H.mH @ Wb
    noise = torch.randn(ya_noiseless.shape, device=ya_noiseless.device,
                        dtype=torch.complex64) * math.sqrt(sigma2)
    ya = ya_noiseless + noise
    # H_est = LA.inv(Wb@Wb.mH)@Wb@ya.mH
    # H_est = Wb@LA.inv(Wb.mH@Wb+sigma2*torch.eye(Ns*num_iter,device=device))@ya.mH
    H_est = LA.inv(Wb@Wb.mH+sigma2*torch.eye(Nb,device=device))@Wb@ya.mH
    mse = LA.norm(H_est-H,dim=(1,2),keepdims=True)/LA.norm(H,dim=(1,2),keepdims=True)
    print('ii=%d, mse=%.4f'%(ii,mse.mean().item()))

    'SVD for estimated H'
    U, S, Vh = torch.linalg.svd(H_est.to('cpu'), full_matrices=False)
    V = Vh.mH
    Wb_est = U[:, :, 0:Ns].to(device)
    Wa_est = V[:, :, 0:Ns].to(device)
    S0 = torch.diag_embed(S[:,0:Ns].to(device))+0j
    rate_lmmse0 = compute_logdet_Y(Wa_est, Wb_est, H)
    print("ii=%d, rate_LMMSE=%.4f" % (ii,rate_lmmse0))
    rate_lmmse = rate_lmmse+rate_lmmse0

    err_a, err_b = compute_dist(Wa_est, Wb_est, Wa0, Wb0)
    print("ii=%d, err_a=%.4f, err_b=%.4f" % (ii, err_a, err_b))
    dist_a, dist_b = err_a+dist_a, err_b+dist_b

rate_opt = rate_opt.item()/num_bsz
rate_lmmse = rate_lmmse.item()/num_bsz
dist_a = dist_a.item()/num_bsz
dist_b = dist_b.item()/num_bsz
print('================================')
print('num_iter=%d'%num_iter)
print('rate_opt=%.4f'%rate_opt)
print("rate_LMMSE=%.4f" %rate_lmmse)

PATH_results = './Results/lmmse_svd_na%d_nb%d_ns%d_iter%d_%ddB' \
             %(Na, Nb, Ns, num_iter, SNR)+'.mat'
sio.savemat(PATH_results, {'rate_opt':rate_opt,'rate_lmmse':rate_lmmse,
                           'bsz':bsz,'num_bsz':num_bsz,'dist_a':dist_a,
                           'dist_b':dist_b})