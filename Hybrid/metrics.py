import torch
import torch.linalg as LA

def compute_rate(Wa, Wb, channel, sigma2_data=0.1):
    Wb = Wb / LA.norm(Wb, dim=[1, 2], keepdim=True)
    Wa = Wa / LA.norm(Wa, dim=[1, 2], keepdim=True)
    He = Wb.mH @ channel @ Wa
    cov_x = He@He.mH
    cov_n = Wb.mH @ Wb * sigma2_data
    rate = torch.log2(LA.det(cov_x+cov_n).real/(LA.det(cov_n).real))
    return torch.mean(rate)


def compute_logdet_Y(Wa, Wb, channel):
    He = Wb.mH @ channel @ Wa
    ns = He.shape[2]
    cov_x = He@He.mH
    cov_x_norm = LA.norm(cov_x,dim=(1,2),keepdims=True)
    cov_x = cov_x/cov_x_norm
    rate = ns*torch.log2(cov_x_norm.squeeze())+torch.log2(LA.det(cov_x).abs())
    return torch.mean(rate)

def compute_mse_LR(Wa, Wb, channel):
    He = Wb.mH @ channel @ Wa
    channel_approx = Wb@He@Wa.mH
    mse = LA.norm(channel_approx-channel,dim=(1,2))/ LA.norm(channel,dim=(1,2))
    return torch.mean(mse)


def compute_mse(Wa, Wb, channel, Hk):
    He = Wb.mH @ channel @ Wa
    channel_approx = Wb@He@Wa.mH
    mse = LA.norm(channel_approx-Hk,dim=(1,2))/ LA.norm(Hk,dim=(1,2))
    return torch.mean(mse)