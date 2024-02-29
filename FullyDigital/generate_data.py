import torch
import math


def generate_channel(bsz, na, nb, random_seed=0):
    if random_seed!=0: torch.manual_seed(random_seed)
    H = torch.randn(bsz, nb, na, dtype=torch.complex64)
    return H


if __name__=='__main__':
    H = generate_channel(bsz=1000, na=64, nb=64)
    HH = H.mH@H@ H.mH@H
    U,S,V = torch.linalg.svd(HH)
    print(S.mean(0))