import torch
import math

def generate_channel(bsz, na, nb, Lp, AoA_range=(-math.pi/3,math.pi/3),
                     random_seed=0):
    phi_min, phi_max = AoA_range
    if random_seed!=0: torch.manual_seed(random_seed)
    phi = (phi_min-phi_max)*torch.rand([bsz,Lp])+phi_max
    sin_phi = torch.sin(phi)
    sin_phi_n = sin_phi[...,None]*torch.arange(na).view(1,1,na)
    a_phi1 = torch.exp(1j*math.pi*sin_phi_n)

    phi = (phi_min-phi_max)*torch.rand([bsz,Lp])+phi_max
    sin_phi = torch.sin(phi)
    sin_phi_n = sin_phi[...,None]*torch.arange(nb).view(1,1,nb)
    a_phi2 = torch.exp(1j*math.pi*sin_phi_n)

    H = a_phi1.view(bsz,Lp,na,1)@a_phi2.view(bsz,Lp,1,nb).conj()
    alpha = torch.randn([bsz, Lp, 1, 1], dtype=torch.complex64)
    H = H*alpha
    H = torch.sum(H,dim=1)/math.sqrt(Lp)
    return H.mH


def generate_channel_grid(bsz, na, nb, Lp, num_grids=256,
                          AoA_range=(-math.pi/3,math.pi/3),random_seed=0):
    phi_min, phi_max = AoA_range
    phi_dict = torch.linspace(phi_min,phi_max,num_grids)
    if random_seed!=0: torch.manual_seed(random_seed)
    for ii in range(bsz):
        idx = torch.randperm(num_grids)[0:Lp]
        idx = idx.sort()[0]
        if ii ==0:
            phi = phi_dict[idx].view(1,Lp)
            idx1 = idx.view(1,Lp)
        else:
            phi = torch.cat([phi,phi_dict[idx].view(1,Lp)],dim=0)
            idx1 = torch.cat([idx1,idx.view(1,Lp)], dim=0)
    sin_phi = torch.sin(phi)
    sin_phi_n = sin_phi[...,None]*torch.arange(na).view(1,1,na)
    a_phi1 = torch.exp(1j*math.pi*sin_phi_n)

    for ii in range(bsz):
        idx = torch.randperm(num_grids)[0:Lp]
        idx = idx.sort()[0]
        if ii ==0:
            phi = phi_dict[idx].view(1,Lp)
            idx2 = idx.view(1,Lp)
        else:
            phi = torch.cat([phi,phi_dict[idx].view(1,Lp)],dim=0)
            idx2 = torch.cat([idx2,idx.view(1,Lp)], dim=0)

    sin_phi = torch.sin(phi)
    sin_phi_n = sin_phi[...,None]*torch.arange(nb).view(1,1,nb)
    a_phi2 = torch.exp(1j*math.pi*sin_phi_n)

    H = a_phi1.view(bsz,Lp,na,1)@a_phi2.view(bsz,Lp,1,nb).conj()
    alpha = torch.randn([bsz, Lp, 1, 1], dtype=torch.complex64)
    H = H*alpha
    H = torch.sum(H,dim=1)/math.sqrt(Lp)
    return H.mH, alpha, idx1, idx2


def generate_dict(na,nb,num_grids=256,AoA_range=(-math.pi/3,math.pi/3)):
    phi_min, phi_max = AoA_range
    phi_dict = torch.linspace(phi_min,phi_max,num_grids).view(1,-1)
    sin_phi = torch.sin(phi_dict)
    sin_phi_n = sin_phi[...,None]*torch.arange(na).view(1,1,na)
    a_phi1 = torch.exp(1j*math.pi*sin_phi_n)

    sin_phi_n = sin_phi[...,None]*torch.arange(nb).view(1,1,nb)
    a_phi2 = torch.exp(1j*math.pi*sin_phi_n)
    return a_phi1[0].T,a_phi2[0].T


if __name__=='__main__':
    generate_channel_grid(bsz=1000, na=32, nb=64,Lp=8)
    a_phi1,a_phi2 = generate_dict(na=32,nb=64,num_grids=256,AoA_range=(
        -math.pi/3,math.pi/3))
