from __init__ import *

class HTV():
    def __init__(self, device):

        self.filters1 = 2 * torch.Tensor([[[[1., -1.,], [-1., 1.]]]]).float().to(device)
        self.filters2 = torch.Tensor([[[[0., 1.], [-1., -1.], [1., 0.]]]]).float().to(device)
        self.filters3 = torch.Tensor([[[[0., -1., 1.], [1., -1., 0.]]]]).float().to(device)

    def L(self, x): 
        Lx1 = F.conv2d(F.pad(x, (1, 1, 1, 1)), self.filters1)
        Lx2 = F.conv2d(F.pad(x, (1, 1, 1, 2)), self.filters2)
        Lx3 = F.conv2d(F.pad(x, (1, 2, 1, 1)), self.filters3)

        Lx = torch.cat((Lx1, Lx2, Lx3), dim=1)

        return Lx
    
    def Lt(self, y):
        Lt1y = F.conv_transpose2d(y[:, :1, :, :], self.filters1)[:, :, 1:-1, 1:-1]
        Lt2y = F.conv_transpose2d(y[:, 1:2, :, :], self.filters2)[:, :, 1:-2, 1:-1]
        Lt3y = F.conv_transpose2d(y[:, 2:3, :, :], self.filters3)[:, :, 1:-1, 1:-2]

        Lty = Lt1y + Lt2y + Lt3y
        
        return Lty
    

if __name__ == '__main__': 

    device = 'cpu'
    x = torch.normal(0, 1, (1, 1, 60, 60)).float().to(device)
    y = torch.normal(0, 1, (1, 3, 61, 61)).float().to(device)
    htv = HTV(device)
    print(htv.L(x).size(), htv.Lt(y).size())
    Hxy = torch.tensordot(htv.L(x), y, 4)
    xHty = torch.tensordot(htv.Lt(y), x, 4)
    print(Hxy, xHty)
    