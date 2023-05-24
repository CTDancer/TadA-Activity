import open3d as o3d
import open3d.t.pipelines.registration as treg

import torch

def calc_struct_sim(p1, p2, cuda=False):

    source = o3d.t.geometry.PointCloud(p1)
    target = o3d.t.geometry.PointCloud(p2)
    if cuda:
        source, target = source.cuda(), target.cuda()

    voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])
    # List of Convergence-Criteria for Multi-Scale ICP:
    criteria_list = [
        treg.ICPConvergenceCriteria(relative_fitness=0.0001,
                                    relative_rmse=0.0001,
                                    max_iteration=20),
        treg.ICPConvergenceCriteria(0.00001, 0.00001, 15),
        treg.ICPConvergenceCriteria(0.000001, 0.000001, 10)
    ]
    max_correspondence_distances = o3d.utility.DoubleVector([128., 64., 32])
    
    registration_ms_icp = treg.multi_scale_icp(
        source, target, voxel_sizes,
        criteria_list,
        max_correspondence_distances,)
    return registration_ms_icp.fitness, registration_ms_icp.inlier_rmse

