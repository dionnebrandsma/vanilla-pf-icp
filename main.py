from vpf import VanillaPFRegistration
import utils

import numpy as np
import open3d as o3d

# Load point cloud
pcd_target = o3d.io.read_point_cloud("data/bun_zipper.ply")
target = np.asarray(pcd_target.points).T

# Downsample 
source = utils.downsample(target, sample_ratio=0.05)
target = utils.downsample(target, sample_ratio=0.05)

# Randomly transform source
source, T_ref = utils.randomly_transformed(source, max_t=0.01, max_theta=20)

# Run PF registration
vpf_reg = VanillaPFRegistration(num_particles=50,
                                max_iter=30,
                                conv_thres=0.005)
aligned, T_est = vpf_reg.register(source, target)

# Compute 6 DOF error
e_theta, e_t = utils.evaluate(T_ref, T_est)

print("Random transformation:\n", T_ref)
print("Estimated transformation:\n", T_est)
print("Error (Rx, Ry, Rz): ", e_theta)
print("Error (tx, ty, tz): ", e_t)

# Visualize result
utils.visualize(target, source, aligned)
