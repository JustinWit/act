import os
import pickle as pkl
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from deoxys_transform_utils import quat2axisangle, axisangle2quat, mat2quat, quat_multiply

APPLY_FIX = True

def get_proprioception(data, gripper_proprio=False):
    if np.isclose(data['gripper_state'].max(), 0.04, atol=1e-2):
        data['gripper_state'] *= 2
    # NOTE 'eef_quat' is "current_quat" which is negated based on "target_quat". We want to use raw robot quats
    raw_quats = [mat2quat(x[:3, :3]) for x in data['eef_pose']]
    if APPLY_FIX:
        axis_angle = np.concatenate([[quat2axisangle(i)] if quat2axisangle(i)[0] > 0.0 else [quat2axisangle(-i)] for i in raw_quats])
    else:
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
        if APPLY_FIX:
            rot = np.array([
                quat2axisangle(quat_multiply(i, j)) if  quat2axisangle(quat_multiply(i, j))[0] > 0.0 else  quat2axisangle(-quat_multiply(i, j)) for i,j in \
                    zip(quat_rot_actions, root['eef_quat'])
                    ])
        else:
            rot = np.array([
                quat2axisangle(quat_multiply(i, j)) for i,j in \
                    zip(quat_rot_actions, root['eef_quat'])
                    ])
        arm_action = np.hstack((pos, rot))
    else:
        arm_action = root['arm_action']
    action = np.hstack((arm_action, root['gripper_action'][:, None]))
    return action


files = [
"datasets/translated_no_cam_rand_slow_fixquat/demo_001.pkl",
"datasets/real_pick_coke/demo_004.pkl",
]

# # pkl_dir = "datasets/translated_no_cam_rand_slow_fixquat"
# pkl_dir = "datasets/real_pick_coke"
# # for i in tqdm(range(500)):
# for i in tqdm(range(4,5)):
for pkl_path in files:
    # pkl_path = os.path.join(pkl_dir, f"demo_{i:03d}.pkl")
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    qpos = get_proprioception(data)
    action = get_action(data, absolute=True)
    # if (action[:, 3] < 0).any():
    # if True:
    # plot both qpos and action. Both have 7 dimensions. Make two plots. Qpos on top with 7 lines, action on bottom with 7 lines
    fig, axs = plt.subplots(2, 1)
    for j in range(3):
        # make the figure tall and skinny so I have lots of vertical resolution
        # fig.set_size_inches(4, 12)
        axs[0].plot(qpos[:, j+3])
        axs[1].plot(action[:, j+3])
        # add titles
        axs[0].set_title('In Proprioception')
        axs[1].set_title('In Actions')
        # add suptitle
        fig.suptitle(f'{pkl_path} Axis-Angle Components')
        # add lines for plus/minus pi
        axs[0].axhline(y=np.pi, color='k', linestyle='--')
        axs[0].axhline(y=-np.pi, color='k', linestyle='--')
        axs[1].axhline(y=np.pi, color='k', linestyle='--')
        axs[1].axhline(y=-np.pi, color='k', linestyle='--')
        # label axes
        axs[0].set_ylabel('Angle (rad)')
        axs[1].set_ylabel('Angle (rad)')
        # axs[0].set_xlabel('timesteps')
        axs[1].set_xlabel('timesteps')

    # add legend
    axs[0].legend([f'{j} axis' for j in ['x', 'y', 'z']])
    axs[1].legend([f'{j} axis' for j in ['x', 'y', 'z']])
    plt.show()
        # exit()
