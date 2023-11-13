from __init__ import * 

class Sampler(): 

    def __init__(self, device):
        
        self.device = device
        self.filter = torch.Tensor([[[[0., 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 0.]]]]).to(device).float() 

        return
    
    def upsample(self, x): 
        # factor = 2
        x_up = F.conv_transpose2d(x, self.filter, padding=1, stride=2)

        return x_up

    def downsample(self, x):
        x_down = F.conv2d(x, self.filter, stride=2, padding=1)    
        return x_down