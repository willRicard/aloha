import time
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env

from constants import DT, START_ARM_POSE, MASTER_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from constants import PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
from robot_utils import Recorder, ImageRecorder
from robot_utils import setup_master_bot, setup_puppet_bot, move_arms, move_grippers
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand

import IPython
e = IPython.embed

class RealEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(self, init_node, setup_robots=True):
        self.puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                                        robot_name=f'puppet_right', init_node=False)
        if setup_robots:
            self.setup_robots()

        self.recorder_right = Recorder('right', init_node=False)
        self.image_recorder = ImageRecorder(init_node=False)
        self.gripper_command = JointSingleCommand(name="gripper")

    def setup_robots(self):
        setup_puppet_bot(self.puppet_bot_right)

    def get_qpos(self):
        right_qpos_raw = self.recorder_right.qpos
        right_arm_qpos = right_qpos_raw[:6]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[7])] # this is position not joint
        return np.concatenate([right_arm_qpos, right_gripper_qpos])

    def get_qvel(self):
        right_qvel_raw = self.recorder_right.qvel
        right_arm_qvel = right_qvel_raw[:6]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[7])]
        return np.concatenate([right_arm_qvel, right_gripper_qvel])

    def get_effort(self):
        right_effort_raw = self.recorder_right.effort
        right_robot_effort = right_effort_raw[:7]
        return np.concatenate([ right_robot_effort])

    def get_images(self):
        return self.image_recorder.get_images()

    def set_gripper_pose(self, right_gripper_desired_pos_normalized):
        right_gripper_desired_joint = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(right_gripper_desired_pos_normalized)
        self.gripper_command.cmd = right_gripper_desired_joint
        self.puppet_bot_right.gripper.core.pub_single.publish(self.gripper_command)

    def _reset_joints(self):
        reset_position = START_ARM_POSE[:6]
        move_arms([self.puppet_bot_right], [reset_position], move_time=1)

    def _reset_gripper(self):
        """Set to position mode and do position resets: first open then close. Then change back to PWM mode"""
        move_grippers([self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)
        move_grippers([self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=1)

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        if not fake:
            # Reboot puppet robot gripper motors
            self.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
            self._reset_joints()
            self._reset_gripper()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def step(self, action):
        right_action = action
        self.puppet_bot_right.arm.set_joint_positions(right_action[:6], blocking=False)
        self.set_gripper_pose(right_action[-1])
        time.sleep(DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())


def get_action(master_bot_right):
    action = np.zeros(7) # 6 joint + 1 gripper, for two arms
    # Arm actions
    action[:6] = master_bot_right.dxl.joint_states.position[:6]
    # Gripper actions
    action[6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])

    return action


def make_real_env(init_node, setup_robots=True):
    env = RealEnv(init_node, setup_robots)
    return env


def test_real_teleop():
    """
    Test bimanual teleoperation and show image observations onscreen.
    It first reads joint poses from both master arms.
    Then use it as actions to step the environment.
    The environment returns full observations including images.

    An alternative approach is to have separate scripts for teleoperation and observation recording.
    This script will result in higher fidelity (obs, action) pairs
    """

    onscreen_render = True
    render_cam = 'cam_high'

    # source of data

    # puppet bot instantiation required here for proper setup 
    _ = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'puppet_right', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    setup_master_bot(master_bot_right)

    # setup the environment
    env = make_real_env(init_node=False)
    ts = env.reset(fake=True)
    episode = [ts]
    # setup visualization
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam])
        plt.ion()

    for t in range(1000):
        action = get_action(master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        if onscreen_render:
            plt_img.set_data(ts.observation['images'][render_cam])
            plt.pause(DT)
        else:
            time.sleep(DT)


if __name__ == '__main__':
    test_real_teleop()
