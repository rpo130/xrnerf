def show_pointcloud_from_depth():
    import copy 
    import numpy as np 
    import open3d as o3d 
    import matplotlib.pyplot as plt

    color = plt.imread('/home/ai/codebase/xrnerf/work_dirs/nerf/nerf_lego_base01/test/001.png')
    depth = np.load('/home/ai/codebase/xrnerf/work_dirs/nerf/nerf_lego_base01/test/001_depth.npy')
    dexdepth = np.load('/home/ai/codebase/xrnerf/work_dirs/nerf/nerf_lego_base01/test/001_dexdepth.npy')



    plt.subplot(131)
    plt.imshow(color)

    plt.subplot(132)
    plt.imshow(depth)

    plt.subplot(133)
    plt.imshow(dexdepth)

    plt.show()

    H,W = depth.shape
    camera_angle_x = 0.6911112070083618
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]


    depth = dexdepth

    rows, cols = depth.shape
    yy,xx = np.meshgrid(np.arange(0, rows), np.arange(0, cols), indexing='ij')
    # xx, yy = np.meshgrid(np.arange(0, cols), np.arange(0, rows))

    xx = xx.flatten()
    yy = yy.flatten()
    zz = depth.flatten()
    x = (xx - cx) * zz / fx
    y = (yy - cy) * zz / fy

    mask = zz > 0
    x = x[mask]
    y = y[mask]
    z = zz[mask]

    points = np.stack([x,y,z], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([pcd])

show_pointcloud_from_depth()