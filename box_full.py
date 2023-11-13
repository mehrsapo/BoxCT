from __init__ import *
from MultiResSolver_ct import *

from skimage.io import imread
from matplotlib import pyplot as plt
import cv2

img = imread('data/ID_0000_AGE_0060_CONTRAST_1_CT.tif') 
img = (img - np.min(img)) / (img.max() - img.min())
image_size = 255
dim = (image_size, image_size)

# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)    
plt.imshow(img, cmap='gray')
plt.colorbar()
plt.show()
    
device = 'cuda:0'
N_grid = 257
N_theta = 360
N_y = 800
thetas = torch.linspace(0, 2 * torch.pi - 0.01, 360).to(device).double()

box_basis = BoxBasis(load=False)
with torch.no_grad():
    box_basis.construct_x_ray_look_up(thetas, name='saved_results/box_sinogram_360.pickle')
    box_basis.construct_x_ray_look_up_torch()

thetas = thetas[:N_theta]
y_values = torch.linspace(-400, 400, N_y)[:, None, None]

c_tensor = F.pad(torch.from_numpy(img), (1, 1, 1, 1))[None, None, :, :].to(device)

x_ray = XrayMS(N_y, N_theta, box_basis, thetas, y_values, device)
x_ray.update_grid(1, N_grid)
y = x_ray.H(c_tensor)

mrs = MultiResSolver('ct', 'htv', lmbda =0, h_init=1, N_scales=1, range_r=256,
                 device=device, verbose=True, toi=1e-6, N_rays=N_y, N_theta=N_theta, box_basis=box_basis, thetas=thetas, y_values=y_values)

mrs.solve_ct(y)

torch.save(mrs.sols[0], 'saved_results/full_rec.pt')