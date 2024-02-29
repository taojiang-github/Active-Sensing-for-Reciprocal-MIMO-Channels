import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Hybrid.generate_data import generate_channel
from model import ActiveSensingFramework
from Hybrid.metrics import compute_logdet_Y
torch.set_num_threads(1)

'System parameter'
Na, Nb, Ns, SNR = 64, 64, 2, 5
sigma2 = 10**(-SNR/10)
num_iter = 8
Lp = 4
N_RF = 8

'Training parameter'
lr, bsz_val, bsz_train, num_epochs, num_batches = 1e-3, 1000, 1024, 1000, 100
initial_run, use_cuda = True, True
device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
print(device)
PATH_model = './param/model_na%d_nb%d_ns%d_iter%d_%ddB_Lp%d_NRF%d' \
             %(Na, Nb, Ns, num_iter, SNR, Lp, N_RF)

'Create model'
hidden_size = 512
model = ActiveSensingFramework(hidden_size, num_iter, Ns, Na, Nb, N_RF).to(device)
if not initial_run: model.load_state_dict(torch.load(PATH_model))
params = model.parameters()
optimizer = torch.optim.Adam(params, lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, eps=1e-5)

'Validation data'
H_val = generate_channel(bsz_val, Na, Nb, Lp, random_seed=1).to(device)
with torch.no_grad():
    Wa, Wb, best_loss_all = model(H_val, sigma2)
    best_loss_val = -compute_logdet_Y(Wa, Wb, H_val)
    U, S, Vh = torch.linalg.svd(H_val.to('cpu'), full_matrices=False)
    V = Vh.mH
    Wb0 = U[:, :, 0:Ns].to(device)
    Wa0 = V[:, :, 0:Ns].to(device)
    S0 = torch.diag_embed(S[:, 0:Ns].to(device)) + 0j
    Hk = Wb0 @ S0 @ Wa0.mH
    rate_opt = compute_logdet_Y(Wa0, Wb0, H_val)
    print("rate_opt=%.4f" % rate_opt)
    print("loss_val=%.4f" % best_loss_val.item())

'Training'
no_increase = 0
for ii in range(num_epochs):
    start_time = time.time()
    loss_train = None
    for jj in range(num_batches):
        H_train = generate_channel(bsz_train, Na, Nb, Lp).to(device)
        Wa, Wb, loss_all = model(H_train,sigma2)
        loss_train = -compute_logdet_Y(Wa, Wb, H_train)
        loss_all.backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        Wa, Wb, loss_all_val = model(H_val, sigma2)
        loss_val = -compute_logdet_Y(Wa, Wb, H_val)
        # if best_loss_all < loss_all_val:
        #     best_loss_all = loss_all_val
        #     no_increase = 0
        #     if ii > 0: torch.save(model.state_dict(), PATH_model)
        # else:
        #     no_increase = no_increase + 1
        if loss_val < best_loss_val:
            best_loss_val = loss_val
            no_increase = 0
            if ii > 5: torch.save(model.state_dict(), PATH_model)
        else:
            no_increase = no_increase + 1
    if no_increase>100: break
    scheduler.step(loss_val)
    print("epoch:%d, lr:%.1e, loss_train:%.4f, loss_val:%.4f,  "
          "best_loss_val:%.4f, gap:%.3f, no_increase:%d, run_time:%.2fsec"
          % (ii, optimizer.param_groups[0]['lr'], loss_train.item(),
             loss_val.item(), best_loss_val.item(),
             rate_opt.item()+loss_val.item(),
             no_increase,(time.time() - start_time)))
