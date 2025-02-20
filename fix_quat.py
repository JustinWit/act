import os
import pickle as pkl
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from deoxys_transform_utils import quat2axisangle, axisangle2quat, mat2quat, quat_multiply


# functions copied from utils.py
def get_proprioception(data, gripper_proprio=False):
    if np.isclose(data['gripper_state'].max(), 0.04, atol=1e-2):
        data['gripper_state'] *= 2
    # NOTE 'eef_quat' is "current_quat" which is negated based on "target_quat". We want to use raw robot quats
    raw_quats = [mat2quat(x[:3, :3]) for x in data['eef_pose']]
    axis_angle = np.concatenate([[quat2axisangle(i)] for i in raw_quats])
    gripper = data['gripper_state'][:, None]
    if not gripper_proprio:
        gripper = np.zeros_like(gripper)
    qpos = np.hstack((data['eef_pos'].squeeze(), axis_angle, gripper))
    return qpos


def get_action(root, absolute=False):
    if absolute:
        pos = root['eef_pos'].squeeze() + root['arm_action'][:, :3]
        quat_rot_actions = [axisangle2quat(x) for x in root['arm_action'][:, 3:]]
        rot = np.array([
            quat2axisangle(quat_multiply(i, j)) for i,j in \
                zip(quat_rot_actions, root['eef_quat'])
                ])
        arm_action = np.hstack((pos, rot))
    else:
        arm_action = root['arm_action']
    action = np.hstack((arm_action, root['gripper_action'][:, None]))
    return action


pkl_dir = "datasets/translated_no_cam_rand_slow_fixquat"
# pkl_dir = "datasets/real_pick_coke"
for i in tqdm(range(500)):
    pkl_path = os.path.join(pkl_dir, f"demo_{i:03d}.pkl")
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    qpos = get_proprioception(data)
    action = get_action(data, absolute=True)
    if (action[:, 3] < 0).any():
        # plot both qpos and action. Both have 7 dimensions. Make two plots. Qpos on top with 7 lines, action on bottom with 7 lines
        fig, axs = plt.subplots(2, 1)
        for j in range(7):
            axs[0].plot(qpos[:, j])
            axs[1].plot(action[:, j])
            # add titles
            axs[0].set_title('qpos')
            axs[1].set_title('action')
            # add suptitle
            fig.suptitle(f'{pkl_dir}/demo_{i:03d}')
        plt.show()
