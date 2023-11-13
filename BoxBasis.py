from sympy import Symbol, integrate, lambdify, simplify, Piecewise
import numpy as np
import torch
import sympytorch
import pickle

def Max(a, b):
    return Piecewise((b, a < b), (a, True))

def Min(a, b):
    return Piecewise((a, a < b), (b, True))

def Box(a):
    cond = ((a <= 1) & (a >= 0))
    return Piecewise((1, cond), (0, True))


class BoxBasis():

    def __init__(self, load=True, name='saved_results/x_ray_look_up_exp_120_pi.pickle'):
        self.y = Symbol('y')
        self.t = Symbol('t')

        if load:
            self.x_ray_look_up = pickle.load(open(name, 'rb'))
            self.construct_x_ray_look_up_torch()
        return
    
    def proj_y(self, theta):
        '''
        returns a function of y for given theta: P_{\theta}\{\varphi\}(y)
        '''
        y = self.y
        t = self.t 

        
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        sin_p_cos_t = sin_t + cos_t
        zero_cos = torch.abs(cos_t) < 1e-6
        zero_sin= torch.abs(sin_t) < 1e-6
        zero_sin_p_cos_t = torch.abs(sin_p_cos_t) < 1e-6

    
        if zero_cos:
            f = Box((y + sin_p_cos_t - t * sin_p_cos_t) * sin_t)

        elif zero_sin:
            f = Box((y + sin_p_cos_t - t * sin_p_cos_t) * cos_t)
        
        elif zero_sin_p_cos_t:
            f = (Max(Min((y - cos_t)/sin_t, 1) - Max((y)/sin_t, 0), 0)) / torch.abs(cos_t)
        
        elif torch.sign(cos_t * sin_t) == 1:
            f = (Max(Min((y + sin_p_cos_t- t * sin_p_cos_t)/sin_t, 1) - Max(((y + sin_p_cos_t - t * sin_p_cos_t - cos_t)/sin_t), 0), 0)) / torch.abs(cos_t)
        else:
            f = (Max(Min((y + sin_p_cos_t - t * sin_p_cos_t - cos_t)/sin_t, 1) - Max((y + sin_p_cos_t - t * sin_p_cos_t)/sin_t, 0), 0)) / torch.abs(cos_t)
        
        out = integrate(f, (t, 0, 1))
            
        return out


    def construct_x_ray_look_up(self, theta_vec, name=None):
        '''
        returns a dictionary that for every theta in theta_vec returns a function P_theta(y)
        ''' 
        self.x_ray_look_up = dict()
        for i, theta in enumerate(theta_vec):
            f_theta =[simplify(self.proj_y(theta))]
            self.x_ray_look_up[i] = f_theta

        if name is not None:
            with open(name, 'wb') as outp:  
                pickle.dump(self.x_ray_look_up, outp)

        return

    def construct_x_ray_look_up_torch(self):
        self.x_ray_look_up_torch = dict()
        with torch.no_grad():
            for key in self.x_ray_look_up:
                self.x_ray_look_up_torch[key] = sympytorch.SymPyModule(expressions=self.x_ray_look_up[key]).requires_grad_(False)

    def x_ray(self, theta, y_vals):
        return self.x_ray_look_up_torch[theta](y=y_vals)