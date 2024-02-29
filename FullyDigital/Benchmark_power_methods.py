import torch
import torch.linalg as LA
import math
from generate_data import generate_channel
from metrics import compute_logdet_Y,compute_dist
import matplotlib.pyplot as plt
import scipy.io as sio
torch.set_grad_enabled(False)

'System parameter'
Na, Nb, Ns, num_iter, SNR = 64, 64, 4, 16, -10
sigma2 = 10**(-SNR/10)

'Algorithm parameter'
use_cuda = True
device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
print(device)
bsz = 1000

'Validation data'
H = generate_channel(bsz, Na, Nb, random_seed=2023).to(device)

# rate_opt = 0.0
U, S, Vh = torch.linalg.svd(H.to('cpu'), full_matrices=False)
V = Vh.mH
Wb0 = U[:, :, 0:Ns].to(device)
Wa0 = V[:, :, 0:Ns].to(device)
# S0 = torch.diag_embed(S[:,0:Ns].to(device))+0j
# Hk = Wb0@S0@Wa0.mH
rate_opt = compute_logdet_Y(Wa0, Wb0, H).item()
print("rate_opt=%.4f" % rate_opt)

'Power method'
Wa = torch.randn(bsz,Na,Ns,device=device)+\
     1j*torch.randn(bsz,Na,Ns,device=device)
Wa = Wa/LA.norm(Wa,dim=1,keepdims=True)
mse_all, rate_all,dist_a, dist_b = [],[],[],[]
for ii in range(num_iter):
    'A transmits to B'
    noise = torch.randn((bsz, Nb, Ns), device=device,
                        dtype=torch.complex64) * math.sqrt(sigma2)
    yb_noiseless = H @ Wa
    yb = yb_noiseless+noise
    Wb,r = LA.qr(yb)

    'B transmits to A'
    noise = torch.randn((bsz, Na, Ns), device=device,
                        dtype=torch.complex64) * math.sqrt(sigma2)
    ya_noiseless = H.mH @ Wb
    ya = ya_noiseless+noise
    Wa,r = LA.qr(ya)

    rate = compute_logdet_Y(Wa, Wb, H)
    rate_all.append(rate.item())

    dist_a0, dist_b0 = compute_dist(Wa, Wb, Wa0, Wb0)
    dist_a.append(dist_a0.item())
    dist_b.append(dist_b0.item())
    # mse = compute_mse(Wa,Wb,H,Hk)
    # mse = rate_opt-rate
    # mse_all.append(mse.item())
    # print('ii=%2d,rate=%.4f,mse=%.4f'%(ii, rate, mse))
    print('ii=%2d,rate=%.4f,err_a=%.4f, err_b=%.4f'%(ii, rate, dist_a0, dist_b0))


plt.figure()
plt.semilogy(torch.arange(num_iter)+1,rate_all,'o-')
plt.grid()
plt.show()

PATH_results = './Results/power_method_na%d_nb%d_ns%d_iter%d_%ddB' \
             %(Na, Nb, Ns, num_iter, SNR)+'.mat'
sio.savemat(PATH_results, {'rate_opt':rate_opt,'rate_power':rate_all,
                           'bsz':bsz,'num_iter':num_iter,'dist_a':dist_a,
                           'dist_b':dist_b})
