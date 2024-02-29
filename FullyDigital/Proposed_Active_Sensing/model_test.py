import time
import torch
import torch.linalg as LA
from FullyDigital.generate_data import generate_channel
from model import ActiveSensingFramework
import matplotlib.pyplot as plt
import math
import scipy.io as sio
from FullyDigital.metrics import compute_rate,compute_logdet_Y,compute_mse
import numpy as np

'System parameter'
Na, Nb, Ns, num_iter_train, SNR = 64, 64, 4, 16, -10
sigma2 = 10**(-SNR/10)

'Algorithm parameter'
bsz = 1000
num_iter_test = 16
use_cuda = True
device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
print(device)
PATH_model = './param/model_na%d_nb%d_ns%d_iter%d_%ddB' \
             %(Na, Nb, Ns, num_iter_train, SNR)

'Create model'
hidden_size = 512
model = ActiveSensingFramework(hidden_size, num_iter_test, Ns, Na, Nb).to(device)
model.load_state_dict(torch.load(PATH_model,map_location=device))
model.eval()

idx = 2

start_time = time.time()
with torch.no_grad():
    H_test = generate_channel(bsz, Na, Nb, random_seed=2023).to(device)

    U, S, Vh = torch.linalg.svd(H_test.to('cpu'), full_matrices=False)
    V = Vh.mH
    Wb0 = U[:, :, 0:Ns].to(device)
    Wa0 = V[:, :, 0:Ns].to(device)
    # S0 = torch.diag_embed(S[:, 0:Ns].to(device)) + 0j
    rate_opt = compute_logdet_Y(Wa0,Wb0,H_test).item()
    print('rate_opt:', rate_opt)

    Wa_all, Wb_all, loss_all = model(H_test, sigma2, True)
    # Wa_final, Wb_final, loss = model(H_test, sigma2, False)

    print('loss_all:', loss_all)
    PATH_results = '../Results/rnn_na%d_nb%d_ns%d_iter%d_%ddB' \
             %(Na, Nb, Ns, num_iter_test, SNR)+'.mat'
    sio.savemat(PATH_results, {'rate_opt':rate_opt,'rate_rnn':loss_all,
                               'bsz':bsz})


   


