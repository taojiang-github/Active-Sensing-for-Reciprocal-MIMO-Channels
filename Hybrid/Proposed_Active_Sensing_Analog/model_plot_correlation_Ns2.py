import time

import numpy as np
import torch
import torch.linalg as LA
from Hybrid.generate_data import generate_channel
from model import ActiveSensingFramework
import matplotlib.pyplot as plt
import math
import scipy.io as sio
from Hybrid.metrics import compute_logdet_Y

def plot_array_response_sensing(Wa_all, Wb_all, U, V, idx=1):
    for ii in range(num_iter):
        if ii==0:
            Wa = Wa_all[ii]
            Wb = Wb_all[ii][idx]
        else:
            Wa = Wa_all[ii][idx]
            Wb = Wb_all[ii][idx]
        response_a=(torch.abs(Wa.mH@V[idx].to(device))).cpu().detach().numpy()
        response_b=(torch.abs(Wb.mH@U[idx].to(device))).cpu().detach().numpy()
        response = response_a

        # if ii==0:
        #     Wa = Wa_all[ii][None,...]
        #     Wb = Wb_all[ii][None,...]
        # else:
        #     Wa = Wa_all[ii]
        #     Wb = Wb_all[ii]
        # response_a=(torch.abs(Wa.mH@V.to(device))).cpu().detach().numpy()
        # response_b=(torch.abs(Wb.mH@U.to(device))).cpu().detach().numpy()
        # response = response_a.mean(0)


        plt.figure()
        linefmt = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        markerfmt = ['o', 's', 'v', '+']
        for jj in range(Ns):
            (markerline, stemlines, baseline)=plt.stem(np.arange(1,1+len(response[jj])),response[jj], linefmt=linefmt[jj], markerfmt = markerfmt[jj], basefmt='black', label='Beamforming vector %d'%(jj+1))
            plt.setp(baseline, visible=False)

        plt.xlabel('Singular vector index',fontsize=14)
        plt.ylabel('Correlation',fontsize=14)
        plt.title('The %dth iteration'%(ii+1),fontsize=14)
        plt.ylim([0, 1])
        plt.legend(loc=1,fontsize=12)
        plt.grid()
        plt.savefig('./fig/sensing_ns%d_SNR_%d_iter_%d'%(Ns,SNR,ii)+'.pdf',bbox_inches='tight')
    plt.show()


def plot_array_response_data(Wa_all, Wb_all, U, V, idx=1):
    Wa = Wa_all
    Wb = Wb_all
    response_a=(torch.abs(Wa.mH@V.to(device))).cpu().detach().numpy()
    response_b=(torch.abs(Wb.mH@U.to(device))).cpu().detach().numpy()

    response = response_a[idx]

    # response = response_a.mean(0)
    plt.figure()
    linefmt = ['tab:blue','tab:orange','tab:green','tab:red']
    markerfmt = ['o','s','v','+']
    for jj in range(Ns):
            (markerline, stemlines, baseline)=plt.stem(np.arange(1,1+len(response[jj])),response[jj], linefmt=linefmt[jj], markerfmt = markerfmt[jj], basefmt='black', label='Beamforming vector %d'%(jj+1))
            plt.setp(baseline, visible=False)
    plt.legend(loc=1,fontsize=12)
    plt.ylim([0,1])
    plt.xlabel('Singular vector index',fontsize=14)
    plt.ylabel('Correlation',fontsize=14)
    # plt.title('Data transmission')
    plt.grid()
    plt.savefig('./fig/data_ns%d_SNR_%d'%(Ns,SNR)+'.pdf',bbox_inches='tight')
    plt.show()

'System parameter'
Na,Nb, Ns, SNR = 64, 64, 2, 5
sigma2 = 10**(-SNR/10)
num_iter = 8
Lp = 4
N_RF = 8

'Algorithm parameter'
bsz = 1000
iter_test = 8
use_cuda = True
device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
print(device)
PATH_model = './param/model_na%d_nb%d_ns%d_iter%d_%ddB_Lp%d_NRF%d' \
             %(Na, Nb, Ns, num_iter, SNR, Lp, N_RF)

'Create model'
hidden_size = 512
model = ActiveSensingFramework(hidden_size, iter_test, Ns, Na, Nb,N_RF).to(device)
model.load_state_dict(torch.load(PATH_model,map_location=device))
model.eval()

start_time = time.time()
with torch.no_grad():
    H_test = generate_channel(bsz, Na, Nb, Lp, random_seed=2023).to(device)

    U, S, Vh = torch.linalg.svd(H_test.to('cpu'), full_matrices=False)
    V = Vh.mH
    Wb0 = U[:, :, 0:Ns].to(device)
    Wa0 = V[:, :, 0:Ns].to(device)
    # S0 = torch.diag_embed(S[:, 0:Ns].to(device)) + 0j
    rate_opt = compute_logdet_Y(Wa0,Wb0,H_test).item()
    print('rate_opt:', rate_opt)

    # Wa, Wb, _ = model(H_test, sigma2, False)
    # rate_rnn = compute_logdet_Y(Wa, Wb, H_test).item()
    # print('rate_rnn:', rate_rnn)
    # PATH_results = './Results/rnn_na%d_nb%d_ns%d_iter%d_%ddB' \
    #              %(Na, Nb, Ns, num_iter, SNR)+'.mat'
    # sio.savemat(PATH_results, {'rate_opt':rate_opt,'rate_rnn':rate_rnn,
    #                            'bsz':bsz})
    
    Wa_all, Wb_all, loss_all = model(H_test, sigma2, True)
    Wa_final, Wb_final, loss = model(H_test, sigma2, False)

    print('loss_all:', loss_all)
    # PATH_results = '../Results/rnn_na%d_nb%d_ns%d_iter%d_%ddB_Lp%d_Nrf%d' \
    #          %(Na, Nb, Ns, num_iter, SNR,Lp,N_RF)+'.mat'
    # sio.savemat(PATH_results, {'rate_opt':rate_opt,'rate_rnn':loss_all,
    #                            'bsz':bsz})

    # plot_array_response_sensing(Wa_all, Wb_all, U, V,idx=2)
    plot_array_response_data(Wa_final, Wb_final, U, V,idx=2)



   


