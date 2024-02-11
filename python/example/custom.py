from matplotlib import pyplot as plt
import matplotlib
import math
from PIL import Image
import imageio
import os

import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym.terrain_utils import *

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Jackal Isaac")

# configure sim
sim_params = gymapi.SimParams()
#sim_params.up_axis = gymapi.UP_AXIS_Z
#sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.static_friction = 100
plane_params.dynamic_friction = 50
gym.add_ground(sim, plane_params)

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
cylinder_asset = gym.create_capsule(sim, 0.075, 1, asset_options)

spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)

envs = []
actor_handles = []
camera_handles = []
depth_handles = []
for i in range(4):
    env = gym.create_env(sim, lower, upper, 2)
    envs.append(env)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(2.25, 0.0, 0.0)
    pose.r = (
        gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), math.pi * -0.5) *
        gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi * -0.5)
    )

    asset_options = gymapi.AssetOptions()

    grid = np.zeros((30, 30)) #np.load(os.path.join(asset_root, "urdf/jackal_description/meshes/grid_0.npy"))
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x][y]:
                xx = x * 0.15; yy = y * 0.15
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(xx, 0.5, yy)
                pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi * -0.5)
                cylinder = gym.create_actor(env, cylinder_asset, pose, "c", i, 0)


'''
arr = 255 - imageio.imread("../../assets/urdf/jackal_description/meshes/test.png")
terrain_width = 10.
terrain_length = 10.
horizontal_scale = 0.03  # [m]
vertical_scale = 0.01  # [m]
num_rows = 332  # int(terrain_width/horizontal_scale)
num_cols = 332  # int(terrain_length/horizontal_scale)
heightfield = arr.astype(np.int16)  # np.zeros((num_rows, num_cols), dtype=np.int16)

terrain = SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
terrain.height_field_raw = heightfield

# add the terrain as a triangle mesh
vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = vertices.shape[0]
tm_params.nb_triangles = triangles.shape[0]
tm_params.transform.p.x = 0
tm_params.transform.p.y = -0.1
tm_params.transform.p.z = 10
tm_params.transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), math.pi * -0.5)
gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)
'''

def tube_trimesh(num_point_ring, vertical_scale, horizontal_scale):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    theta_grid = np.linspace(-np.pi, np.pi, num_point_ring)
    heights = 1 - np.cos(theta_grid)
    hf = np.stack([heights] * 20, axis=-1)

    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.sin(theta_grid) * vertical_scale
    yy, xx = np.meshgrid(y, x)

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles

vertices, triangles = tube_trimesh(10, 1, 0.2)
tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = vertices.shape[0]
tm_params.nb_triangles = triangles.shape[0]
#tm_params.transform.p.x = 0
#tm_params.transform.p.y = -0.1
#tm_params.transform.p.z = 10
#tm_params.transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), math.pi * -0.5)
gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

for k in range(100000000):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    #gym.sync_frame_time(sim)




print("Initialized Jackal robot")
