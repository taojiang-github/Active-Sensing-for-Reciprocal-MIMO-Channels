import torch
import torch.nn as nn
import torch.linalg as LA
import math
from Hybrid.metrics import compute_logdet_Y,compute_rate


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
        self.dense1 = nn.Linear(dim_in, 512)
        self.dense2 = nn.Linear(512, n * ns * 2)
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
    def __init__(self, hsz, n_stages, ns, na, nb, nrf):
        super(ActiveSensingFramework, self).__init__()
        self.cell_a = nn.GRUCell(input_size=2 * nrf, hidden_size=hsz)
        self.cell_b = nn.GRUCell(input_size=2 * nrf, hidden_size=hsz)
        self.mlp_a_W = MLPLayers(hsz, nrf)
        self.mlp_b_W = MLPLayers(hsz, nrf)
        self.mlp_a_final = MLPLayers(hsz, nrf)
        self.mlp_b_final = MLPLayers(hsz, nrf)

        self.linear_a = nn.Linear(hsz, 128)
        self.linear_b = nn.Linear(hsz, 128)
        self.mlp_a_Fr = MLPLayers(128 * ns, na, nrf)
        self.mlp_b_Fr = MLPLayers(128 * ns, nb, nrf)
        self.mlp_a_Ft = MLPLayers(128 * ns, na, nrf)
        self.mlp_b_Ft = MLPLayers(128 * ns, nb, nrf)

        self.hsz = hsz
        self.n_stages = n_stages
        self.ns = ns
        self.na = na
        self.nrf = nrf

        self.init_Wa = torch.nn.Parameter(torch.randn(nrf, ns, dtype=torch.complex64),requires_grad=True)
        self.init_Fb = torch.nn.Parameter(torch.randn(nb, nrf, dtype=torch.complex64),requires_grad=True)
        self.init_Fa = torch.nn.Parameter(torch.randn(na, nrf, dtype=torch.complex64),requires_grad=True)

    def forward(self, channel, sigma2, return_sensing=False):
        bsz, nb, na = channel.shape
        hs_a, hs_b = [], []
        for ii in range(self.ns):
            hs_a.append(torch.ones(bsz, self.hsz, device=channel.device))
            hs_b.append(torch.ones(bsz, self.hsz, device=channel.device))
        Fb = self.init_Fb.to(channel.device)
        Fb = Fb/(math.sqrt(nb)*Fb.abs())

        Wa = self.init_Wa.to(channel.device)
        Fa = self.init_Fa.to(channel.device)
        Fa = Fa/(math.sqrt(na)*Fa.abs())
        FWa = Fa@Wa
        FWa = FWa/LA.norm(FWa,dim=0,keepdims=True)

        Wa_all, Wb_all, loss_all, loss = [FWa],[],[], 0.0
        for ii in range(self.n_stages):
            'A transmits to B'
            yb = channel@FWa
            noise = torch.randn(yb.shape, device=yb.device, dtype=torch.complex64)*math.sqrt(sigma2)
            yb = yb+noise
            yb = Fb.mH@yb
            hs_b_stack = []
            for jj in range(self.ns):
                y_input = nn.Flatten()(torch.view_as_real(yb[:,:,jj]))
                hs_b[jj] = self.cell_b(y_input, hs_b[jj])
                hs_b_stack.append(self.linear_b(hs_b[jj]))
                if jj==0:
                    Wb0 = self.mlp_b_W(hs_b[jj])
                else:
                    Wb0 = torch.cat([Wb0,self.mlp_b_W(hs_b[jj])], dim=2)
            Wb = orthogonal(Wb0+yb, self.ns)
            hs_b_stack = torch.stack(hs_b_stack,-1)
            Fb = self.mlp_b_Ft(nn.Flatten()(hs_b_stack))
            Fb = Fb/(math.sqrt(nb)*Fb.abs())
            FWb = Fb @ Wb
            FWb = FWb / LA.norm(FWb, dim=1, keepdims=True)
            Wb_all.append(FWb)

            Fb = self.mlp_b_Fr(nn.Flatten()(hs_b_stack))
            Fb = Fb/(math.sqrt(nb)*Fb.abs())

            'B transmits to A'
            ya = channel.mH@FWb
            noise = torch.randn(ya.shape, device=ya.device,dtype=torch.complex64)*math.sqrt(sigma2)
            ya = ya+noise
            ya = Fa.mH@ya
            hs_a_stack = []
            for jj in range(self.ns):
                y_input = nn.Flatten()(torch.view_as_real(ya[:,:,jj]))
                hs_a[jj]= self.cell_a(y_input, hs_a[jj])
                hs_a_stack.append(self.linear_a(hs_a[jj]))
                if jj==0:
                    Wa0 = self.mlp_a_W(hs_a[jj])
                else:
                    Wa0 = torch.cat([Wa0,self.mlp_a_W(hs_a[jj])], dim=2)
            Wa = orthogonal(Wa0+ya, self.ns)
            hs_a_stack = torch.stack(hs_a_stack,-1)
            Fa = self.mlp_a_Ft(nn.Flatten()(hs_a_stack))
            Fa = Fa/(math.sqrt(na) * Fa.abs())
            FWa = Fa @ Wa
            FWa = FWa / LA.norm(FWa, dim=1, keepdims=True)
            Wa_all.append(FWa)

            Fa = self.mlp_a_Fr(nn.Flatten()(hs_a_stack))
            Fa = Fa / (math.sqrt(na) * Fa.abs())

            'design the final data transmission beamformer'
            for jj in range(self.ns):
                if jj == 0:
                    Wb0 = self.mlp_b_final(hs_b[jj])
                    Wa0 = self.mlp_a_final(hs_a[jj])
                else:
                    Wb0 = torch.cat([Wb0, self.mlp_b_final(hs_b[jj])], dim=2)
                    Wa0 = torch.cat([Wa0, self.mlp_a_final(hs_a[jj])], dim=2)

            Wb_final = Fb@orthogonal(Wb0+yb, self.ns)
            Wa_final = Fa@orthogonal(Wa0+ya, self.ns)

            Wb_final = Wb_final/LA.norm(Wb_final,dim=1,keepdims=True)
            Wa_final = Wa_final/LA.norm(Wa_final,dim=1,keepdims=True)

            loss_i = -compute_logdet_Y(Wa_final, Wb_final, channel)
            # loss_i = -compute_rate(Wa_final, Wb_final, channel)
            loss = loss+loss_i
            loss_all.append(-loss_i.item())
        if return_sensing:
            return Wa_all, Wb_all, loss_all
        else:
            return Wa_final, Wb_final, loss


