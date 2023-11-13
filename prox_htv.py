from __init__ import * 

class prox_htv(): 

    def __init__(self, device):
         
        self.device = device
        self.htv = HTV(device)

    
    def eval(self, y, N, niter, lmbda, verbose=False, toi=1e-4):

        v_k = torch.zeros((1, 3, N+1, N+1), requires_grad=False, device=self.device).float()
        u_k = torch.zeros((1, 3, N+1, N+1), requires_grad=False, device=self.device).float()

        t_k = 1 

        alpha = 1 / 64

        self.loss_list = list()

        for _ in range(niter):
            LTv = self.htv.Lt(v_k)
            Lpc = self.htv.L(torch.clip(y - LTv, 0, 1))

            if verbose: 
                loss = ((y - torch.clip(y - LTv, 0, 1))**2).sum() / 2 + lmbda * torch.norm(Lpc, dim=1, p=1).sum()
                self.loss_list.append(loss.item())
                
            u_kp1 = torch.clip(v_k + alpha * Lpc, -lmbda , lmbda)
            t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
            v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)                                                                                                                                                                                                                                                                                                                                                         

            err_rel = torch.norm(u_k - u_kp1, p='fro') / max((torch.norm(u_k, p='fro')).item(), 1e-10)

            u_k = u_kp1
            v_k = v_kp1
            t_k = t_kp1
   
            if err_rel < toi: 
                break


        c = y - self.htv.Lt(u_k)
        c = torch.clip(c, 0, 1)

        return c
