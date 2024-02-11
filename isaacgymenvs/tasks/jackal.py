from turtle import pd
import numpy as np
import os, time
import imageio
import math
from copy import deepcopy
from matplotlib import pyplot as plt
import matplotlib

from isaacgym.torch_utils import *
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.terrain_utils import convert_heightfield_to_trimesh

from .base.vec_task import VecTask

import torch
from typing import Any, Tuple, Dict

@torch.jit.script
def depth_image_to_point_cloud_GPU(
    camera_tensor,
    camera_view_matrix_inv,
    camera_proj_matrix,
    width:float, height:float,
    device:torch.device
):
    camera_u = torch.arange(0, width, device=device)
    camera_v = torch.arange(0, height, device=device)
    v, u = torch.meshgrid(camera_v, camera_u, indexing='ij')

    depth_buffer = camera_tensor.to(device)
    vinv = camera_view_matrix_inv
    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv

    Z = Z.view(-1)
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))
    position = position.permute(1, 0)
    position = position@vinv

    points = position[:, 0:3]

    return points

def print_tensor_info(tensor, name):
    print(f"{name}:")
    print(f"  Type: {tensor.dtype}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Device: {tensor.device}")
    print(f"  Memory (approx): {tensor.element_size() * tensor.nelement() / (1024**2):.2f} MB\n")

# # Example usage after your environment setup
# print_tensor_info(self.root_states, "Root States")
# print_tensor_info(self.dof_state, "DOF State")
# print_tensor_info(self.contact_forces, "Contact Forces")
# # Add similar lines for other tensors you're interested in

class Jakcal(VecTask):

    def __init__(
        self, cfg, rl_device, sim_device,
        graphics_device_id, headless,
        virtual_screen_capture: bool = False,
        force_render: bool = False
    ):
        self.cfg = cfg
        self.height_samples = None
        self.custom_origins = False
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.init_done = False
        # initial RL related propoerties here
        # pass

        # base init state
        self.start_pos = self.cfg["env"]["baseInitState"]["pos"]
        self.start_rot = self.cfg["env"]["baseInitState"]["rot"]
        goal = self.cfg["env"]["baseInitState"]["goal"]
        self.goal_origin = torch.tensor([[goal[0], goal[1], 0]] * self.cfg["env"]["numEnvs"], device=sim_device)
        self.goal = deepcopy(self.goal_origin)
        self.max_action = torch.tensor(np.array([-1e6, -1e6]), device= sim_device)
        self.min_action = torch.tensor(np.array([1e6, 1e6]), device = sim_device)
        # other
        self.decimation = self.cfg["env"]["control"]["decimation"]  # number of skipped steps
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s/ self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.spacing = self.cfg["env"]["envSpacing"]

        super().__init__(
            cfg, rl_device, sim_device,
            graphics_device_id, headless,
            virtual_screen_capture, force_render
        )

        if self.graphics_device_id != -1:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        # (num_actors, 13)
        # position[0:3], rotation[3:7], linear_velocity[7:10], angular_velocity[10:13]
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # DOF state might not be used
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)

        # (num_actors, 2)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # .view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)

        self.root_states_start = deepcopy(self.root_states)
        self.last_goal_dist = deepcopy(torch.linalg.norm(self.root_states[self.jackal_actor_idx][:, :2] - self.goal_origin[:, :2], dim=-1))
        self.actions = torch.zeros((self.cfg["env"]["numEnvs"], self.cfg["env"]["numActions"]), device=self.device)

        # initialize some data used later on
        # Check whether we actually need it

        # reward episode sums
        # Check whether we actually need it

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.init_done = True
        print_tensor_info(self.root_states, "Root States")
        print_tensor_info(self.dof_state, "DOF State")
        print_tensor_info(self.contact_forces, "Contact Forces")

    def create_sim(self):
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()

        self._create_envs(self.num_envs)

    def _create_envs(self, num_envs):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = self.cfg["env"]["urdfAsset"]["file"]
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        # asset_options.flip_visual_attachments = True
        # asset_options.density = 0.001
        asset_options.override_inertia = True
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.005
        asset_options.disable_gravity = False

        jackal_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        # set wheel friction of Jackal
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(jackal_asset)
        for rsp in rigid_shape_prop:
            rsp.friction = 0.5

        # set Jackal DOF properties
        dof_props = self.gym.get_asset_dof_properties(jackal_asset)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_VEL)
        dof_props["stiffness"].fill(self.Kp)
        dof_props["damping"].fill(self.Kd)

        # setup cylinders for BARN worlds
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        cylinder_asset = self.gym.create_capsule(self.sim, 0.075, 0.5, asset_options)
        box_asset_side = self.gym.create_box(self.sim, 0.15 * 60, 0.15, 1, asset_options)
        box_asset_back = self.gym.create_box(self.sim, 0.15 * 30, 0.15, 1, asset_options)

        self.num_dof = self.gym.get_asset_dof_count(jackal_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(jackal_asset)

        num_jackal_bodies = self.gym.get_asset_rigid_body_count(jackal_asset)
        num_jackal_shapes = self.gym.get_asset_rigid_shape_count(jackal_asset)
        num_cylinder_bodies = self.gym.get_asset_rigid_body_count(cylinder_asset)
        num_cylinder_shapes = self.gym.get_asset_rigid_shape_count(cylinder_asset)
        num_box_bodies = self.gym.get_asset_rigid_body_count(box_asset_side)
        num_box_shapes = self.gym.get_asset_rigid_shape_count(box_asset_side)

        env_lower = gymapi.Vec3(-self.spacing/2, -self.spacing, 0)
        env_upper = gymapi.Vec3(self.spacing/2, self.spacing, self.spacing)
        self.jackal_handles = []
        self.camera_handles_envs = []
        self.envs = []
        cylinder_count_total = 0
        self.jackal_actor_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.jackal_rigid_body_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        print_tensor_info(self.jackal_actor_idx, "Jackal Actor Idx")
        print_tensor_info(self.jackal_rigid_body_idx, "Jackal Rigid Body Idx")
        map_folder = self.cfg["env"]["terrain"]["mapFolder"]
        grid_root = os.path.join(asset_root, f"../{map_folder}")
        grid_files = [f for f in os.listdir(grid_root) if f.endswith(".npy")]
        grid_files = np.random.permutation(grid_files * self.num_envs)[:self.num_envs]
        self.worlds = ["world_%s" %(f.split("_")[-1].split(".")[0]) for f in grid_files]
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            grid = np.load(os.path.join(grid_root, grid_files[i]))
            max_agg_bodies = num_jackal_bodies + np.sum(grid[1:-1, 1:-1]) * num_cylinder_bodies + num_box_bodies * 3
            max_agg_shapes = num_jackal_shapes + np.sum(grid[1:-1, 1:-1]) * num_cylinder_shapes + num_box_shapes * 3

            self.jackal_actor_idx[i] = cylinder_count_total + i
            self.jackal_rigid_body_idx[i] = cylinder_count_total + i * 5  # the main body is the first rigid body
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*self.start_pos)
            pose.r = (
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi * self.start_rot / 180)
            )

            # aggregate will cause free(): invalid pointer bug
            # self.gym.begin_aggregate(env_handle, max_agg_bodies, max_agg_shapes, True)
            jackal_handle = self.gym.create_actor(env_handle, jackal_asset, pose, "jackal", i, 0, 0)
            self.gym.set_actor_dof_properties(env_handle, jackal_handle, dof_props)

            
            cylinder_count = 3
            
            for x in range(1, grid.shape[0]-1):  # remove the wall and replace it with box
                for y in range(1, grid.shape[1]-1):
                    if grid[x][y]:
                        cylinder_count += 1
                        xx = x * 0.15; yy = y * 0.15 + 0.15 * 30
                        pose = gymapi.Transform()
                        pose.p = gymapi.Vec3(xx, yy, 0.5)
                        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), math.pi * 0.5)
                        cylinder = self.gym.create_actor(env_handle, cylinder_asset, pose, "cylinder_%d" %(cylinder_count), i, 1, 0)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0.15 * 30, 0.5)
            pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi * 0.5)
            box_side_1 = self.gym.create_actor(env_handle, box_asset_side, pose, "box_side_1", i, 1, 0)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.15 * 29, 0.15 * 30, 0.5)
            pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi * 0.5)
            box_side_2 = self.gym.create_actor(env_handle, box_asset_side, pose, "box_side_2", i, 1, 0)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.15 * 14.5, 0, 0.5)
            pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi * 0)
            box_back = self.gym.create_actor(env_handle, box_asset_back, pose, "box_back", i, 1, 0)
            
            # aggregate will cause free(): invalid pointer bug
            # self.gym.end_aggregate(env_handle)
            cylinder_count_total += cylinder_count

            self.envs.append(env_handle)
            self.jackal_handles.append(jackal_handle)

            camera_handles = []
            for angle in [-90, 0, 90]:
                camera_props = gymapi.CameraProperties()
                camera_props.enable_tensors = True
                camera_props.width = self.cfg["env"]["camera"]["width"]
                camera_props.height = self.cfg["env"]["camera"]["height"]
                camera_props.horizontal_fov = self.cfg["env"]["camera"]["horizontal_fov"]
                camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                local_transform = gymapi.Transform()
                local_transform.p = gymapi.Vec3(0.12, 0, 0.3)  # check the location of this camera
                local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(angle))
                self.gym.attach_camera_to_body(camera_handle, env_handle, jackal_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
                camera_handles.append(camera_handle)
            self.camera_handles_envs.append(camera_handles)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        obs, rew, done, info = super().step(actions)
        info["success"] = self.success_buf
        info["collision"] = self.collided_buf
        info["world"] = self.worlds
        info["time_outs"] = self.timeout_buf_tmp
        info["dof_vel"] = self.dof_vel
        info["dof_pos"] = self.dof_pos
        return obs, rew, done, info

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        plane_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def check_termination(self):
        self.reset_buf = torch.logical_or(torch.logical_or(self.timeout_buf_tmp, self.collided_buf), self.success_buf)

    def compute_observation(self):
        self.gym.start_access_image_tensors(self.sim)
        camera_tensors_envs = []
        for e, che in zip(self.envs, self.camera_handles_envs):
            lasers = []
            for i, ch in enumerate(che):
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, e, ch, gymapi.IMAGE_DEPTH)
                camera_tensor = gymtorch.wrap_tensor(camera_tensor)

                cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, e, ch)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, e, ch), device=self.device)
                cam_vinv[-1, :2] = 0
                points_cam = depth_image_to_point_cloud_GPU(
                    camera_tensor, cam_vinv, cam_proj, 240, 1, self.device
                )
                laser = torch.linalg.norm(points_cam[:, :2], dim=1)
                laser = torch.nan_to_num(laser, nan=10.0)  # max is 10 meter
                laser = torch.clip(laser, max=10.0)
                lasers.append(laser / 10.0)  # rough normalization
                """
                points_plots = points_cam.detach().cpu().numpy()
                plt.scatter(points_plots[:100, 0], points_plots[:100, 1], c="red")
                plt.scatter(points_plots[100:, 0], points_plots[100:, 1], c="blue")
                plt.xlim([-8, 8])
                plt.ylim([-8, 8])
                """
                '''
                # For debug use, save the image as this
                camera_tensor_tmp = self.gym.get_camera_image_gpu_tensor(self.sim, e, ch, gymapi.IMAGE_COLOR)
                camera_tensor_tmp = gymtorch.wrap_tensor(camera_tensor_tmp)
                print("shape", camera_tensor_tmp.shape)
                # import pdb; pdb.set_trace()
                matplotlib.image.imsave('images/depth_%d.png' %(i), torch.clip(-camera_tensor, min=0., max=20).detach().cpu())
                matplotlib.image.imsave('images/color_%d.png' %(i), camera_tensor_tmp.detach().cpu().numpy())
                '''
            # plt.show()
            
            camera_tensors_envs.append(torch.cat(lasers[::-1], dim=0))
        self.gym.end_access_image_tensors(self.sim)
        laser = torch.stack(camera_tensors_envs, dim=0)
        '''
        from matplotlib import pyplot as plt
        img = np.zeros((256, 256))
        print(laser[0, :: 4])
        laser = torch.nan_to_num(laser, nan=10.0)
        laser = torch.clip(laser, max=10.0)
        for i, l in enumerate(laser[0, :]):
            theta = -3/4*np.pi + 3/2*np.pi * (i / 720.)
            x = int(l * np.cos(theta) * 128 / 10.) + 127
            y = int(l * np.sin(theta) * 128 / 10.) + 127
            img[x, y] = 1
        plt.imshow(img)
        plt.show()
        '''
        goal_pos = self.goal - self.root_states[self.jackal_actor_idx][:, :3]  # goal position in world frame
        psi = self.root_states[self.jackal_actor_idx][:, 3:7]  # quaternion in world frame
        psi[:, :-1] = -psi[:, :-1]  # theta to -theta
        goal_robot = self.qrot(psi, goal_pos)  # goal position in the robot frame
        self.obs_buf = torch.cat([laser, goal_robot[:, :2] / 10., self.actions], dim=-1)  # this 10.0 is a very rough normalization
        # self.obs_buf = torch.cat([laser, goal_pos[:, :2] / 10., psi[:, -2:]], dim=-1)

    def qrot(self, q, v):
        """
        Copied from https://github.com/facebookresearch/QuaterNet/blob/main/common/quaternion.py
        Rotate vector(s) v about the rotation described by quaternion(s) q.
        Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4
        assert v.shape[-1] == 3
        assert q.shape[:-1] == v.shape[:-1]
        
        original_shape = list(v.shape)
        q = q.view(-1, 4)
        v = v.view(-1, 3)
        
        qvec = q[:, :-1]
        uv = torch.cross(qvec, v, dim=1)
        uuv = torch.cross(qvec, uv, dim=1)
        return (v + 2 * (q[:, -1:] * uv + uuv)).view(original_shape)
    
    def compute_reward(self):
        # self.collided_buf = torch.linalg.norm(self.contact_forces[self.jackal_rigid_body_idx][:, :2], dim=-1) > 0.01
        self.timeout_buf_tmp = self.progress_buf >= self.max_episode_length
        self.success_buf = torch.linalg.norm(self.root_states[self.jackal_actor_idx][:, :2] - self.goal[:, :2], dim=-1) < 1

        self.rew_buf = torch.zeros_like(self.rew_buf)
        self.rew_buf += self.success_buf * self.cfg["env"]["learn"]["success_reward"]
        self.rew_buf += self.collided_buf * self.cfg["env"]["learn"]["collision_reward"]

        current_goal_dist = torch.linalg.norm(self.root_states[self.jackal_actor_idx][:, :2] - self.goal[:, :2], dim=-1)
        self.rew_buf += (-current_goal_dist + self.last_goal_dist) * self.cfg["env"]["learn"]["progress_reward"]

        self.last_goal_dist = current_goal_dist
        
    def reset_idx(self, env_ids):
        actor_ids = self.jackal_actor_idx[env_ids]
        self.root_states[actor_ids] = self.root_states_start[actor_ids]
        self.root_states[actor_ids][:, :2] += torch.rand(self.root_states[actor_ids][:, :2].shape).to(self.device) * 0.5
        rot = torch.rand(len(actor_ids)).to(self.device) * torch.pi * 2 - torch.pi
        self.root_states[actor_ids][:, 5:7] = torch.stack([torch.cos(rot), torch.sin(rot)], dim=1)
        actor_ids = actor_ids.to(torch.int32)
        #env_ids_int32 = env_ids.to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(actor_ids), len(actor_ids)
        )
        # import pdb; pdb.set_trace()
        #self.gym.set_dof_state_tensor_indexed(
        #    self.sim,
        #    gymtorch.unwrap_tensor(self.dof_state),
        #    gymtorch.unwrap_tensor(actor_ids), len(actor_ids)
        #)
        self.progress_buf[env_ids] = 0
        #self.reset_buf[env_ids] = 0
        self.goal[env_ids] = deepcopy(self.goal_origin[env_ids])
        self.goal[env_ids] += torch.rand(self.goal[env_ids].shape).to(self.device) * 0.5
        self.last_goal_dist[env_ids] = torch.linalg.norm(self.root_states[self.jackal_actor_idx][env_ids, :2] - self.goal[env_ids, :2], dim=-1)

        # if an actor is reset, force to update the camera and contact force tensor
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.step_graphics(self.sim)
    
    def pre_physics_step(self, actions):
        # actions: (num_envs, num_actions)
        # action is vx and w
        # mapping from (vx, w) to w_R and w_L
        # w_R = (w*b + 2*v_x) / (2r); w_L = (-w*b + 2*v_x) / (2r)
        # b = 0.37559 (robot width); r = 0.098 (wheel radius)
        # How the actions are translated and applied
        self.collided_buf = torch.linalg.norm(self.contact_forces[self.jackal_rigid_body_idx][:, :3], dim=-1) > 0.01
        self.actions = actions.clone().to(self.device) * self.cfg["env"]["control"]["actionScale"]

        # Since the angular velocity does not match the actual, we need a multiplier
        # this range varies, we randomize it in a range
        # get the actions range
        # # Compute the max and min values across the specified dimension
        # max_values, _ = torch.max(self.actions, dim=0)
        # min_values, _ = torch.min(self.actions, dim=0)

        # # Update self.max_action and self.min_action
        # self.max_action = torch.max(self.max_action, max_values)
        # self.min_action = torch.min(self.min_action, min_values)
        # # print("max_action", self.max_action)
        # # print("min_action", self.min_action)

        multiplier = np.random.uniform(*self.cfg["env"]["control"]["multiplier"])
        actions[:, 0] = actions[:, 0] * 2; actions[:, 1] = actions[:, 1] * 3.14 * multiplier
        for i in range(self.decimation):  # repeat this action for self.decimation frames
            wR = (2 * actions[:, 0] + actions[:, 1] * 0.37559) / (2 * 0.098)
            wL = (2 * actions[:, 0] - actions[:, 1] * 0.37559) / (2 * 0.098)
            vel_targets = gymtorch.unwrap_tensor(torch.stack([wR, wL], axis=1).repeat(1, 2))
            self.gym.set_dof_velocity_target_tensor(self.sim, vel_targets)

            self.gym.simulate(self.sim)
            # if self.device == 'cpu':
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)

            # check collision at every sim step rather than env step
            collided_buf = torch.linalg.norm(self.contact_forces[self.jackal_rigid_body_idx][:, :3], dim=-1) > 0.01
            collided_buf = torch.logical_or(collided_buf, self.root_states[self.jackal_actor_idx][:, 2] > 0.15)
            self.collided_buf = torch.logical_or(collided_buf, self.collided_buf)
            self.collided_buf = torch.logical_and(self.collided_buf, self.progress_buf > 1)

        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        
    def post_physics_step(self):
        self.progress_buf += 1
        self.compute_reward()
        self.check_termination()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observation()


class RslRLJackal(Jakcal):
    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)

        # self.num_obs = self.num_observations
        self.num_privileged_obs = None
        self.privileged_obs_buf = None
        self.episode_length_buf = self.progress_buf
    
    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self, *args, **kw_args):
        self.episode_length_buf = self.progress_buf
        obs = super().reset(*args, **kw_args)
        return obs["obs"], self.privileged_obs_buf

    def step(self, *args, **kw_args):
        self.episode_length_buf = self.progress_buf
        obs, rew, done, info = super().step(*args, **kw_args)
        return obs["obs"], self.privileged_obs_buf, rew, done, info