import torch
import torch.linalg as LA
import math
from FullyDigital.generate_data import generate_channel
from FullyDigital.metrics import compute_logdet_Y,compute_dist
import matplotlib.pyplot as plt
import scipy.io as sio
torch.set_grad_enabled(False)
import numpy as np

torch.manual_seed(2023)
np.random.seed(2023)

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

def plot_array_response_sensing(Wa_all, Wb_all, U, V, idx):
    for ii in range(num_iter):
        Wa = torch.tensor(Wa_all[idx][ii])
        Wb = torch.tensor(Wb_all[idx][ii])
        response_a=(torch.abs(Wa.mH@V[idx])).cpu().detach().numpy()
        response_b=(torch.abs(Wb.mH@U[idx])).cpu().detach().numpy()
        response = response_a

        # Wa = torch.tensor(Wa_all[:,ii,:,:])
        # Wb = torch.tensor(Wb_all[:,ii,:,:])
        # response_a=(torch.abs(Wa.mH@V)).cpu().detach().numpy()
        # response_b=(torch.abs(Wb.mH@U)).cpu().detach().numpy()
        # response = response_a.mean(0)

        plt.figure()
        linefmt = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        markerfmt = ['o', 's', 'v', '+']
        for jj in range(Ns):
            (markerline, stemlines, baseline)=plt.stem(np.arange(1,1+len(response[jj])),response[jj], linefmt=linefmt[jj], markerfmt = markerfmt[jj], basefmt='black', label='Beamforming vector %d'%(jj+1))
            plt.setp(baseline, visible=False)

        plt.xlabel('Singular vector index',fontsize=14)
        plt.ylabel('Correlation',fontsize=14)
        # plt.title('The %dth iteration'%(ii+1),fontsize=14)
        plt.ylim([0, 0.8])
        plt.legend(loc=1,fontsize=12)
        plt.grid()
        plt.savefig(dir_path+'/array_response/power_sensing_ns%d_SNR_%d_iter_%d'%(Ns,SNR,ii)+'.pdf',bbox_inches='tight')
    plt.show()

'System parameter'
Na, Nb, Ns, num_iter, SNR = 64, 64, 4, 16, 0
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

Wa_all = np.zeros([bsz,num_iter+1, Na,Ns],dtype=np.complex64)
Wb_all = np.zeros([bsz,num_iter+1, Nb,Ns],dtype=np.complex64)

'Power method'
Wa = torch.randn(bsz,Na,Ns,device=device)+\
     1j*torch.randn(bsz,Na,Ns,device=device)
Wa = Wa/LA.norm(Wa,dim=1,keepdims=True)
Wa_all[:, 0, :, :] =  Wa.cpu().detach().numpy()

mse_all, rate_all,dist_a, dist_b = [],[],[],[]
for ii in range(num_iter):
    'A transmits to B'
    noise = torch.randn((bsz, Nb, Ns), device=device,
                        dtype=torch.complex64) * math.sqrt(sigma2)
    yb_noiseless = H @ Wa
    yb = yb_noiseless+noise
    Wb,r = LA.qr(yb)

    Wb_all[:, ii, :, :] = Wb.cpu().detach().numpy()


    'B transmits to A'
    noise = torch.randn((bsz, Na, Ns), device=device,
                        dtype=torch.complex64) * math.sqrt(sigma2)
    ya_noiseless = H.mH @ Wb
    ya = ya_noiseless+noise
    Wa,r = LA.qr(ya)

    Wa_all[:, ii, :, :] = Wa.cpu().detach().numpy()

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


# plt.figure()
# plt.semilogy(torch.arange(num_iter)+1,rate_all,'o-')
# plt.grid()
# plt.show()

# PATH_results = './Results/power_method_na%d_nb%d_ns%d_iter%d_%ddB' \
#              %(Na, Nb, Ns, num_iter, SNR)+'.mat'
# sio.savemat(PATH_results, {'rate_opt':rate_opt,'rate_power':rate_all,
#                            'bsz':bsz,'num_iter':num_iter,'dist_a':dist_a,
#                            'dist_b':dist_b})
plot_array_response_sensing(Wa_all, Wb_all, U, V,idx=6)