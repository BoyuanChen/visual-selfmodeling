
import os
import math
import json
import time
import fusion
import datetime
import pybullet
import numpy as np
import pybullet_data
from utils import common
from fusion import TSDFVolume
import pybullet_utils.bullet_client as bc


class ArmPybulletSim(object):
    def __init__(self, gui_enabled, num_cam):

        # robot viewable range
        self._view_bounds = np.array([[-0.45, 0.45],
                                      [-0.45, 0.45],
                                      [ 0.000, 0.84]]) # 3x2 rows: x,y,z cols: min,max. hard


        self._volume_bounds = self._view_bounds
        self._voxel_size = 0.005

        if gui_enabled:
            self.bullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.bullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self.bullet_client.resetSimulation()
        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bullet_client.setGravity(0, 0, -9.8)

        self._plane_id = self.bullet_client.loadURDF("plane.urdf")
        self.bullet_client.changeDynamics(self._plane_id, -1, lateralFriction=1.0)

        # add RGB-D camera (mimic RealSense D435i)
        self._scene_cam_lookat = self._view_bounds.mean(axis=1)
        self._scene_cam_position = [self._scene_cam_lookat[0], self._scene_cam_lookat[1], 0.5]
        self._scene_cam_up_direction = [0, 0, 1]
        self._scene_cam_image_size = (480, 640)
        self._scene_cam_z_near = 0.01
        self._scene_cam_z_far = 10.0
        self._scene_cam_fov_w = 69.40
        self._scene_cam_focal_length = (float(self._scene_cam_image_size[1])/2)/np.tan((np.pi*self._scene_cam_fov_w/180)/2)
        self._scene_cam_fov_h = (math.atan((float(self._scene_cam_image_size[0])/2)/self._scene_cam_focal_length)*2/np.pi)*180
        self._scene_cam_projection_matrix = self.bullet_client.computeProjectionMatrixFOV(
            fov=self._scene_cam_fov_h,
            aspect=float(self._scene_cam_image_size[1])/float(self._scene_cam_image_size[0]),
            nearVal=self._scene_cam_z_near, farVal=self._scene_cam_z_far
        )
        self._scene_cam_intrinsics = np.array([[self._scene_cam_focal_length, 0, float(self._scene_cam_image_size[1])/2],
                                             [0, self._scene_cam_focal_length, float(self._scene_cam_image_size[0])/2],
                                             [0, 0, 1]])

        self._num_cam = num_cam
        self.debug_cam = False
        self._num_joints = 5
        self._n_steps = 0
        self._timeout = 1 # in seconds
        self._saved_robot_state_dict = {}
        self._ctrl_mode = pybullet.POSITION_CONTROL
        self.debug_cam_sphere_urdf = "assets/widowx_arm_description/urdf/weightless_sphere.urdf"
        self._robot_urdf = "assets/interbotix_descriptions/urdf/wx200_gripper.urdf"
        self.save_mesh_folder = "saved_meshes"
        common.mkdir(self.save_mesh_folder)

    # Get latest RGB-D image from scene camera
    def get_scene_cam_data(self, cam_position=None, cam_lookat=None, cam_up_direction=None):
        if cam_position is None:
            cam_position = self._scene_cam_position
        if cam_lookat is None:
            cam_lookat = self._scene_cam_lookat
        if cam_up_direction is None:
            cam_up_direction = self._scene_cam_up_direction
        cam_view_matrix = self.bullet_client.computeViewMatrix(cam_position, cam_lookat, cam_up_direction)

        cam_pose_matrix = np.linalg.inv(np.array(cam_view_matrix).reshape(4, 4).T)
        cam_pose_matrix[:, 1:3] = -cam_pose_matrix[:, 1:3]

        camera_data = self.bullet_client.getCameraImage(
            self._scene_cam_image_size[1],
            self._scene_cam_image_size[0],
            cam_view_matrix,
            self._scene_cam_projection_matrix,
            shadow=1,
            flags=self.bullet_client.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_pixels = np.array(camera_data[2]).reshape((self._scene_cam_image_size[0], self._scene_cam_image_size[1], 4))
        color_image = rgb_pixels[:,:,:3] # remove alpha channel
        z_buffer = np.array(camera_data[3]).reshape((self._scene_cam_image_size[0], self._scene_cam_image_size[1]))
        segmentation_mask = np.array(camera_data[4], np.int) # - not implemented yet with renderer=p.ER_BULLET_HARDWARE_OPENGL
        depth_image = (2.0*self._scene_cam_z_near*self._scene_cam_z_far)/(self._scene_cam_z_far+self._scene_cam_z_near-(2.0*z_buffer-1.0)*(self._scene_cam_z_far-self._scene_cam_z_near))
        return color_image, depth_image, segmentation_mask, cam_pose_matrix

    def reset(self):
        # load robot
        robot_start_pos = [0, 0, 0.01]
        robot_start_ori = pybullet.getQuaternionFromEuler([0, 0, 0])
        self._robot_id = self.bullet_client.loadURDF(self._robot_urdf, robot_start_pos, robot_start_ori, useFixedBase=True, flags=self.bullet_client.URDF_USE_SELF_COLLISION)

        # return observation: robot state, scene observation
        observation = self._get_observation(mesh_idx=0, if_save=False)
        return observation
    
    def reset_everything(self):
        self.bullet_client.resetSimulation()
        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bullet_client.setGravity(0, 0, -9.8)
        
        self._plane_id = self.bullet_client.loadURDF("plane.urdf")
        self.bullet_client.changeDynamics(self._plane_id, -1, lateralFriction=1.0)

        robot_start_pos = [0, 0, 0.01]
        robot_start_ori = pybullet.getQuaternionFromEuler([0, 0, 0])
        self._robot_id = self.bullet_client.loadURDF(self._robot_urdf, robot_start_pos, robot_start_ori)


    def _get_observation(self, mesh_idx, if_save):
        scene_observation = self._get_scene_observation(mesh_idx, if_save)

        # get the robot state as part of the observation: joint position and velocity
        robot_state_lst = []
        for i in range(self._num_joints):
            joint_pos, joint_vel, _, _ = self.bullet_client.getJointState(self._robot_id, i)
            robot_state_lst.append([joint_pos, joint_vel])
        
        # get the cartesian value of the last link
        last_link_position = self.bullet_client.getLinkState(self._robot_id, 5, 1, 1)[0]
        robot_state_lst = robot_state_lst + [last_link_position]

        observation = {
            'robot_state': robot_state_lst,
            'scene_observation': scene_observation
        }

        return observation

    def _get_scene_observation(self, mesh_idx, if_save):

        # remove the plane to obtain the robot mesh
        visual_data = dict()
        visual_data[self._plane_id] = common.get_body_colors(self.bullet_client, self._plane_id)
        common.set_visible(self.bullet_client, visual_data, visible=False)

        self._scene_tsdf = TSDFVolume(self._volume_bounds, voxel_size=self._voxel_size)

        # surrounding cameras
        scene_center = self._view_bounds.mean(axis=1)
        cam_positions = [[-1.0, -1.0, 0.42],
                         [-1.0, 1.0, 0.42],
                         [1.0, -1.0, 0.42],
                         [1.0, 1.0, 0.42]] # hard
        for cam_id in range(self._num_cam-1):
            cam_position = cam_positions[cam_id]
            cam_lookat = [scene_center[0], scene_center[1], 0.42] #hard
            cam_up_direction = [0, 0, 1]
            if self.debug_cam:
                _ = self.bullet_client.loadURDF(self.debug_cam_sphere_urdf, cam_position, pybullet.getQuaternionFromEuler([0,0,0]), )
            color_image, depth_image, segmentation_mask, cam_pose_matrix = self.get_scene_cam_data(cam_position, cam_lookat, cam_up_direction)
            self._scene_tsdf.integrate(color_image, depth_image, self._scene_cam_intrinsics, cam_pose_matrix, obs_weight=1.)

        # top-down camera
        cam_position = [scene_center[0], scene_center[1], 1.5] # hard
        cam_lookat = [scene_center[0], scene_center[1], 0]
        cam_up_direction = [0, 1, 0]
        if self.debug_cam:
            _ = self.bullet_client.loadURDF(self.debug_cam_sphere_urdf, cam_position, pybullet.getQuaternionFromEuler([0,0,0]), )
        color_image, depth_image, segmentation_mask, cam_pose_matrix = self.get_scene_cam_data(cam_position, cam_lookat, cam_up_direction)
        self._scene_tsdf.integrate(color_image, depth_image, self._scene_cam_intrinsics, cam_pose_matrix, obs_weight=1.)

        # get scene_tsdf(WxHxD) and obstacle_vol(WxHxDx3)
        scene_tsdf, obstacle_vol = self._scene_tsdf.get_volume()

        if if_save:
            verts, faces, norms, colors = self._scene_tsdf.get_mesh()
            fusion.meshwrite(os.path.join(self.save_mesh_folder, f"mesh_{mesh_idx}.ply"), verts, faces, norms, colors)

        scene_observation = {
            'scene_tsdf': scene_tsdf,
            'obstacle_vol': obstacle_vol,
        }

        # bring the plane back
        common.set_visible(self.bullet_client, visual_data, visible=True)

        return scene_observation

    def step(self, action):

        reward = None
        done = None
        info = None

        forces = np.array([5, 5, 5, 5, 5])

        if action['joint'] is None:
            for i in range(len(action['robot'])):
                self.bullet_client.setJointMotorControl2(self._robot_id,
                                                         i,
                                                         controlMode=self._ctrl_mode,
                                                         targetPosition=action['robot'][i],
                                                         force=forces[i])
        else:
            for p_joint in action['joint']:
                self.bullet_client.setJointMotorControl2(self._robot_id,
                                                        p_joint,
                                                        controlMode=self._ctrl_mode,
                                                        targetPosition=action['robot'][p_joint],
                                                        force=forces[p_joint])
            for i in range(len(action['robot'])):
                if i not in action['joint']:
                    self.bullet_client.setJointMotorControl2(self._robot_id,
                                                             i,
                                                             controlMode=self._ctrl_mode,
                                                             targetPosition=0.,
                                                             force=forces[i])

        # wait until the arm becomes stable or timeout (due to unreachable state)
        success_flag = False
        start_time = datetime.datetime.now()
        while True:
            self.bullet_client.stepSimulation()
            if self._if_robot_stable() and self._if_reach_target_joint(action):
                success_flag = True
                break
            passed_time = (datetime.datetime.now() - start_time).total_seconds()
            if passed_time > self._timeout:
                success_flag = False
                break

        # return observation: robot state, scene observation
        if success_flag:
            observation = self._get_observation(mesh_idx=self._n_steps, if_save=True)
            self._saved_robot_state_dict[self._n_steps] = observation['robot_state']
            self._n_steps = self._n_steps + 1
        else:
            observation = None
        return observation, reward, done, info

    def save_robot_state(self):
        with open(os.path.join(self.save_mesh_folder, 'robot_state.json'), 'w') as file:
            json.dump(self._saved_robot_state_dict, file, indent=4)

    def _if_robot_stable(self):
        joint_vel_lst = []
        for i in range(self._num_joints):
            _, joint_vel, _, _ = self.bullet_client.getJointState(self._robot_id, i)
            joint_vel_lst.append(joint_vel)
        if np.abs(np.sum(joint_vel_lst)) <= 0.005:
            return True
        else:
            return False

    def _if_reach_target_joint(self, action):
        for i in range(self._num_joints):
            joint_pos, joint_vel, _, _ = self.bullet_client.getJointState(self._robot_id, i)
            # if all the joints are being controlled.
            if action['joint'] is None:
                joint_diff = joint_pos - action['robot'][i]
                if np.abs(joint_diff) > 1e-3:
                    return False
            # if only specific joints are being controlled.
            else:
                if i in action['joint']:
                    joint_diff = joint_pos - action['robot'][i]
                else:
                    joint_diff = joint_pos - 0.
                if np.abs(joint_diff) > 1e-3:
                    return False
        return True

if __name__ == '__main__':

    env = ArmPybulletSim(gui_enabled=False, num_cam=5)
    N = 10000
    while True:
        env.reset_everything()
        action = {'robot': np.random.uniform(-1, 1, env._num_joints) * np.pi, 'joint': [0, 1, 2, 3]}
        obs, r, done, _ = env.step(action)
        print(env._n_steps)
        if env._n_steps == N:
            break
    env.save_robot_state()


