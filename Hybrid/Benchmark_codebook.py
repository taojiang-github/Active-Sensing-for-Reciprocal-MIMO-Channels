import numpy as np
import scipy.io as sio
from generate_data import generate_channel
import torch
import torch.linalg as LA
from metrics import compute_logdet_Y
from scipy.linalg import dft

def generate_beamformer(Na, k, ii, N_RF):
    Ma = N_RF*2**(k-1)
    if ii>=Ma: raise Exception("ii <= Ma-1")
    phi_a_i = np.pi*(1-(2*ii+1)/Ma)
    w = np.zeros(Na)+1j*np.zeros(Na)
    w[0:Ma] = np.exp(-1j*phi_a_i*np.arange(Ma))
    w = w/np.linalg.norm(w)
    return w[:,None]


def generate_beamformer_sets(Na, k, N_RF):
    for ii in range(N_RF):
        if ii==0:
            w = generate_beamformer(Na, k, ii, N_RF)
        else:
            w = np.concatenate([w,generate_beamformer(Na, k, ii, N_RF)],axis=1)
    return w


def update_analog_precoder(F, W, p, k, i, K):
    row_norm = -np.linalg.norm(W,axis=1)
    p = p+list(row_norm)
    N = F.shape[0]
    pI = np.argsort(p)
    bI, bL = [], []
    n= 0
    while n<N_RF:
        bI.append(pI[n])
        p[pI[n]] = 0
        if k[pI[n]]==K or n==N_RF-1:
            bL.append(1)
            n = n+1
        else:
            bL.append(2)
            n = n+2
    F = []
    k_new = []
    i_new = []
    for t in range(len(bI)):
        m = bL[t]
        if m == 2:
            w1 = generate_beamformer(N,k[bI[t]]+1,i[bI[t]]*2,N_RF)
            w2 = generate_beamformer(N,k[bI[t]]+1,i[bI[t]]*2+1,N_RF)
            F.append(w1[:,0])
            F.append(w2[:,0])
            k_new.append(k[bI[t]]+1)
            k_new.append(k[bI[t]]+1)
            i_new.append(i[bI[t]]*2)
            i_new.append(i[bI[t]]*2+1)
            # F = np.concatenate([F,w1,w2],axis=1)
        else:
            w = generate_beamformer(N,k[bI[t]],i[bI[t]],N_RF)
            F.append(w[:,0])
            k_new.append(k[bI[t]])
            i_new.append(i[bI[t]])
            # F = np.concatenate([F,w],axis=1)
    return np.array(F).T, p, k+k_new, i+i_new


def compute_objective(Wa,Wb,Fa,Fb,G):
    Wa0 = Fa@Wa
    Wa0 = Wa0/np.linalg.norm(Wa0,axis=0,keepdims=True)
    Wa0 = torch.tensor(Wa0,dtype=torch.complex64)
    Wb0 = Fb@Wb
    Wb0 = Wb0/np.linalg.norm(Wb0,axis=0,keepdims=True)
    Wb0 = torch.tensor(Wb0,dtype=torch.complex64)
    obj = compute_logdet_Y(Wa0[None,...],Wb0[None,...],G[None,...])
    return obj


'System parameter'
Na, Nb, Ns, SNR = 64, 64, 4, 10
sigma2 = 10**(-SNR/10)
num_iter = 8
Lp = 10
N_RF = 8
KA = np.log2(Na/N_RF)+1
KB = np.log2(Nb/N_RF)+1

'Algorithm parameter'
use_cuda = False
device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
print(device)
bsz = 1000

'Validation data'
H = generate_channel(bsz, Na, Nb, Lp, random_seed=2023).to(device)
U, S, Vh = torch.linalg.svd(H.to('cpu'), full_matrices=False)
V = Vh.mH
Wb0 = U[:, :, 0:Ns].to(device)
Wa0 = V[:, :, 0:Ns].to(device)
rate_opt = compute_logdet_Y(Wa0, Wb0, H)
print("rate_opt=%.4f" % rate_opt)

obj_all = np.zeros(num_iter)
for ii in range(bsz):
    G = H[ii].numpy(force=True)
    Fa = generate_beamformer_sets(Na,1,N_RF)
    Fb = generate_beamformer_sets(Nb,1,N_RF)

    Wa = dft(N_RF)[:,0:Ns]
    # Wa = np.random.randn(N_RF,Ns)+1j*np.random.randn(N_RF,Ns)
    Wa = Wa/np.linalg.norm(Fa@Wa,axis=0,keepdims=True)
    pa, ka, ia = [], [1]*N_RF, list(np.arange(N_RF))
    pb, kb, ib = [], [1]*N_RF, list(np.arange(N_RF))

    'A transmits to B'
    yb_noiseless = G @ Fa @ Wa
    noise = np.random.normal(loc=0, scale=np.sqrt(0.5), size=np.shape(
        yb_noiseless)) + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5),
                                               size=np.shape(yb_noiseless))
    yb = yb_noiseless + noise * np.sqrt(sigma2)
    yb = Fb.T.conj() @ yb
    Qb, R = np.linalg.qr(yb)
    Wb = Qb / np.linalg.norm(Fb @ Qb, axis=0, keepdims=True)
    obj_i = []
    for jj in range(num_iter):
        'B transmits to A'
        ya_noiseless = G.T.conj()@Fb@Wb
        noise = np.random.normal(loc=0, scale=np.sqrt(0.5), size=np.shape(ya_noiseless)) \
            + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=np.shape(ya_noiseless))
        ya = ya_noiseless+noise*np.sqrt(sigma2)
        ya = Fa.T.conj()@ya
        Qa, R = np.linalg.qr(ya)
        Wa = Qa/np.linalg.norm(Fa@Qa,axis=0,keepdims=True)

        'B updates analog beamformer'
        Fb, pb, kb, ib = update_analog_precoder(Fb, Wb, pb, kb, ib, KB)

        'A transmits to B'
        yb_noiseless = G @ Fa @ Wa
        noise = np.random.normal(loc=0, scale=np.sqrt(0.5), size=np.shape(
            yb_noiseless)) + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5),
                                                   size=np.shape(yb_noiseless))
        yb = yb_noiseless + noise * np.sqrt(sigma2)
        yb = Fb.T.conj() @ yb
        Qb, R = np.linalg.qr(yb)
        Wb = Qb / np.linalg.norm(Fb @ Qb, axis=0, keepdims=True)

        'A updates analog beamformer'
        Fa, pa, ka, ia = update_analog_precoder(Fa, Wa, pa, ka, ia, KA)

        obj = compute_objective(Wa,Wb,Fa,Fb,H[ii])
        obj_i.append(obj.item())
        # print('jj=%d, obj=%f'%(jj, obj))
    obj_all = obj_all+np.array(obj_i)
    if ii%100==0 and ii>0:
        obj_all_avg = obj_all/(ii+1)
        print('ii=%d, bf_gain_opt=%.4f'%(ii, rate_opt))
        print(obj_all_avg)

        PATH_results = './Results/codebook_power_method_na%d_nb%d_ns%d_iter%d_' \
                       '%ddB_Lp%d_NRF%d' % (Na, Nb, Ns, num_iter, SNR, Lp, N_RF) + '.mat'
        sio.savemat(PATH_results,
                    {'rate_opt': rate_opt.item(), 'rate_power': obj_all_avg,
                     'bsz': bsz, 'num_iter': num_iter})