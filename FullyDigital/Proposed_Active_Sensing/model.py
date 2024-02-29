import torch
import torch.nn as nn
import torch.linalg as LA
from torch.autograd import Variable
import math
from FullyDigital.metrics import compute_logdet_Y


def proj(u,v):
    return (u.mH@v/(u.mH@u))*u


def orthogonal(x, ns):
    for ii in range(ns):
        x_i = torch.unsqueeze(x[:, :, ii], -1)
        for jj in range(ii):
            x_j = torch.unsqueeze(x[:, :, jj], -1)
            x_i = x_i - proj(x_j, x_i)
        if ii == 0:
            x_i = x_i / LA.norm(x_i, dim=1, keepdim=True)
            x_new = x_i
        else:
            x_i = x_i / LA.norm(x_i, dim=1, keepdim=True)
            x_new = torch.cat([x_new, x_i], dim=2)
    return x_new


class MLPLayers(nn.Module):
    def __init__(self, dim_in, n, ns=1):
        super(MLPLayers, self).__init__()
        self.dense1 = nn.Linear(dim_in, 1024)
        self.dense2 = nn.Linear(1024, n * ns * 2)
        self.dim_out = (n, ns)

    def forward(self, x):
        (n, ns) = self.dim_out
        x = self.dense2(torch.relu(self.dense1(x)))
        x_real = x[:, 0:n * ns]
        x_imag = x[:, n * ns:2 * ns * n]
        x = torch.complex(x_real, x_imag)
        x = torch.reshape(x, shape=(-1, n, ns))
        return x


class ActiveSensingFramework(nn.Module):
    def __init__(self, hsz, n_stages, ns, na, nb):
        super(ActiveSensingFramework, self).__init__()
        self.cell_a = nn.GRUCell(input_size=2 * na, hidden_size=hsz)
        self.cell_b = nn.GRUCell(input_size=2 * nb, hidden_size=hsz)
        Wa_init = torch.randn(na, ns, dtype=torch.complex64,requires_grad=True)
        Wa_init = Wa_init / LA.norm(Wa_init, dim=0, keepdims=True)
        self.init_Wa = torch.nn.Parameter(Wa_init,requires_grad=True)
        self.mlp_a = MLPLayers(hsz, na)
        self.mlp_b = MLPLayers(hsz, nb)
        self.mlp_a1 = MLPLayers(hsz, na)
        self.mlp_b1 = MLPLayers(hsz, nb)
        self.hsz = hsz
        self.n_stages = n_stages
        self.ns = ns
        self.na = na

    def forward(self, channel, sigma2, return_sensing=False):
        bsz, nb, na = channel.shape
        hs_a, hs_b = [], []
        for ii in range(self.ns):
            hs_a.append(torch.ones(bsz, self.hsz, device=channel.device))
            hs_b.append(torch.ones(bsz, self.hsz, device=channel.device))
        Fa_tx = self.init_Wa.to(channel.device)
        ns = Fa_tx.shape[1]
        Fa_tx = Fa_tx/LA.norm(Fa_tx,dim=0,keepdims=True)
        Fa_tx_all, Fb_tx_all, loss_all, loss = [Fa_tx],[],[],0.0
        for ii in range(self.n_stages):
            'A transmits to B'
            y_noiseless = channel@Fa_tx
            noise = torch.randn((bsz,nb,ns), device=y_noiseless.device,
                                dtype=torch.complex64)*math.sqrt(sigma2)
            yb = y_noiseless+noise
            for jj in range(self.ns):
                y_input = nn.Flatten()(torch.view_as_real(yb[:,:,jj]))
                hs_b[jj] = self.cell_b(y_input, hs_b[jj])
                if jj==0:
                    Fb_tx0 = self.mlp_b(hs_b[jj])
                else:
                    Fb_tx0 = torch.cat([Fb_tx0,self.mlp_b(hs_b[jj])], dim=2)
            Fb_tx = orthogonal(Fb_tx0+yb, self.ns)
            Fb_tx_all.append(Fb_tx)

            'B transmits to A'
            y_noiseless = channel.mH@Fb_tx
            noise = torch.randn((bsz,na,ns), device=y_noiseless.device,
                                dtype=torch.complex64)*math.sqrt(sigma2)
            ya = y_noiseless+noise
            for jj in range(self.ns):
                y_input = nn.Flatten()(torch.view_as_real(ya[:,:,jj]))
                hs_a[jj]= self.cell_a(y_input, hs_a[jj])
                if jj==0:
                    Fa_tx0 = self.mlp_a(hs_a[jj])
                else:
                    Fa_tx0 = torch.cat([Fa_tx0,self.mlp_a(hs_a[jj])], dim=2)
            Fa_tx = orthogonal(Fa_tx0+ya, self.ns)
            Fa_tx_all.append(Fa_tx)

            'data transmission beamformer'
            for jj in range(self.ns):
                if jj == 0:
                    Fb_tx0 = self.mlp_b1(hs_b[jj])
                    Fa_tx0 = self.mlp_a1(hs_a[jj])
                else:
                    Fb_tx0 = torch.cat([Fb_tx0, self.mlp_b1(hs_b[jj])], dim=2)
                    Fa_tx0 = torch.cat([Fa_tx0, self.mlp_a1(hs_a[jj])], dim=2)
            Fb_tx = orthogonal(Fb_tx0+yb, self.ns)
            Fa_tx = orthogonal(Fa_tx0+ya, self.ns)

            loss_i = -compute_logdet_Y(Fa_tx, Fb_tx, channel)
            loss = loss+loss_i
            loss_all.append(-loss_i.item())
        if return_sensing:
            return Fa_tx_all, Fb_tx_all, loss_all
        else:
            return Fa_tx, Fb_tx, loss


