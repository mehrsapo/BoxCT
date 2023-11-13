from BoxBasis import  *
import torch


class XrayMS():

    def __init__(self, N_y, N_theta, box_basis, thetas, y_values, device):
        self.box_basis = box_basis
        self.N_y = N_y
        self.N_theta = N_theta
        self.device = device

        self.thetas = thetas
        self.y_values = y_values.to(device)
        

    def update_grid(self, T, N_grid):
        self.T = T
        self.N_grid = N_grid
        with torch.no_grad():
            self.k_1 = torch.arange(0, N_grid, 1).repeat((N_grid, 1)).to(self.device).float()
            self.k_2 = torch.arange(N_grid-1, -1, -1).repeat((N_grid, 1)).to(self.device).transpose(1, 0)
        
    def H(self, c):
        with torch.no_grad():
            full_sinogram = torch.zeros((self.N_theta, self.N_y)).to(self.device)
            for i, theta in enumerate(self.thetas):
                itorchut_arg = (self.y_values / self.T - self.k_1 * torch.cos(theta) - self.k_2 * torch.sin(theta))
                H_y_theta = self.T * self.box_basis.x_ray(i, itorchut_arg).transpose(3, 1) 
                full_sinogram[i:i+1, :] = (H_y_theta * c).sum(dim=(2, 3)) .transpose(1, 0).float()
        return full_sinogram

    
    def Ht(self, d):
        with torch.no_grad():
            fbp = torch.zeros((1, 1, self.N_grid, self.N_grid)).to(self.device).float()
            for i, theta in enumerate(self.thetas):
                itorchut_arg = (self.y_values / self.T - self.k_1 * torch.cos(theta) - self.k_2 * torch.sin(theta))
                H_y_theta = self.T * self.box_basis.x_ray(i, itorchut_arg).transpose(3, 1)
                fbp +=  (d[i][:, None, None, None] * H_y_theta.float()).sum(dim=(0), keepdim=True)
        return fbp