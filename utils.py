import numpy as np
import torch
import os
import h5py
import pickle as pkl
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from deoxys.utils.transform_utils import quat2axisangle

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, proprioception=True, chunk_size=None, all_demos=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.use_proprio = proprioception
        # self.__getitem__(0) # initialize self.is_sim
        self.is_sim = False
        self.chunk_size = chunk_size
        self.all_demos = all_demos

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode
        if self.all_demos is None:
            # use f-string to format index to be three digits with leading zeros
            episode_id = f'{self.episode_ids[index]:03d}'
            dataset_path = os.path.join(self.dataset_dir, f'demo_{episode_id}.pkl')
            with open(dataset_path, 'rb') as db:
                root = pkl.load(db)
        else:
            root = self.all_demos[index]

        # is_sim = root.attrs['sim']
        action = np.hstack((root['arm_action'], root['gripper_action'][:, None]))
        original_action_shape = action.shape
        episode_len = original_action_shape[0]
        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)
        # get observation at start_ts only
        if self.use_proprio:
            if np.isclose(root['gripper_action'].max(), 0.04, atol=1e-2):
                root['gripper_action'] = root['gripper_action'] * 2
            pos = root['eef_pos'].squeeze()[start_ts]
            axis_angle = quat2axisangle(root['eef_quat'][start_ts])
            gripper = root['gripper_action'][:, None][start_ts]
            qpos = np.concatenate((pos, axis_angle, gripper))
        else:
            qpos = np.zeros(7)
        # qvel = root['/observations/qvel'][start_ts]
        # image_dict = dict()
        for cam_name in self.camera_names:
            if root[cam_name].shape[1] == 3:  # real data has 3 cams
                cam_image = root[cam_name][start_ts][2]
                assert cam_image.shape == (360, 640, 3)
            elif root[cam_name].shape[1] == 1:  # sim data has one cam
                cam_image = root[cam_name][start_ts][0]
                assert cam_image.shape == (256, 256, 3)
            else:
                raise ValueError('Unknown camera shape')
            # convert bgr to rgb
            cam_image = cam_image[..., ::-1]
            if cam_image.shape == (360, 640, 3):
                # center crop to 360 x 360
                cam_image = cam_image[:, 140: 500]
                # downsize to 256 x 256
                cam_image = cv2.resize(cam_image, (256, 256))
                # cam_image = np.array(Image.fromarray(cam_image).resize((256, 256)))
            else:
                assert cam_image.shape == (256, 256, 3)
            cam_image = cam_image[None, :]
        # plt.imsave('img.png', cam_image[0])

        # get all actions after and including start_ts
        action_len = episode_len - start_ts
        action = action[start_ts:]
        # if is_sim:
        #     action = root['/action'][start_ts:]
        #     action_len = episode_len - start_ts
        # else:
        #     action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
        #     action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras NOTE: this is not necessary for our case we have single cam
        # all_cam_images = []
        # for cam_name in self.camera_names:
        #     all_cam_images.append(cam_image)
        # all_cam_images = np.stack(all_cam_images, axis=0)
        # breakpoint()

        # construct observations
        image_data = torch.from_numpy(cam_image.copy())
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last -> channel first
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        # added this since the actions just get truncated by the model anyways
        action_data = action_data[:self.chunk_size]
        is_pad = is_pad[:self.chunk_size]
        return image_data.float(), qpos_data.float(), action_data.float(), is_pad


def get_norm_stats(
    dataset_dir,
    num_episodes,
    proprioception=True,
    preload_data=False,
    ):
    all_qpos_data = []
    all_action_data = []
    all_demos = []
    listdir = [x for x in os.listdir(dataset_dir) if x.endswith('.pkl')]
    for i, episode_path in enumerate(tqdm(listdir, desc='Loading data')):
        # dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with open(os.path.join(dataset_dir, episode_path), 'rb') as dbfile:
            root = pkl.load(dbfile)
        if preload_data:
            all_demos.append(root)
        # get action and pose data
        # qpos = root['/observations/qpos'][()]
        if np.isclose(root['gripper_action'].max(), 0.04, atol=1e-2):
            root['gripper_action'] = root['gripper_action'] * 2
        print(root['gripper_action'][:, None].shape)
        # check quat to axisangle
        axis_angle = np.concatenate([[quat2axisangle(i)] for i in root['eef_quat']])
        qpos = np.hstack((root['eef_pos'].squeeze(), axis_angle, root['gripper_state'][:, None]))
        # qvel = root['/observations/qvel'][()]
        action = np.hstack((root['arm_action'], root['gripper_action'][:, None]))
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    all_qpos_data = torch.vstack(all_qpos_data)
    if not proprioception:
        all_qpos_data = torch.zeros_like(all_qpos_data)
    all_action_data = torch.vstack(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = all_action_data.std(dim=[0], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats, all_demos


def collate_fn(batch):
    image_data, qpos_data, action_data, is_pad = zip(*batch)

    image_data = torch.stack(image_data)
    qpos_data = torch.stack(qpos_data)
    # need to add padding so that all sequences have the same length
    action_data = torch.nn.utils.rnn.pad_sequence(action_data, padding_value=0, batch_first=True)
    is_pad = torch.nn.utils.rnn.pad_sequence(is_pad, padding_value=1, batch_first=True)
    batch = (image_data, qpos_data, action_data, is_pad)
    return batch


def load_data(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    proprioception=True,
    chunk_size=None,
    preload_data=False,
    ):

    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 1.0
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats, all_demos = get_norm_stats(
        dataset_dir,
        num_episodes,
        proprioception=proprioception,
        preload_data=preload_data,
        )

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        train_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        proprioception=proprioception,
        chunk_size=chunk_size,
        all_demos=all_demos if preload_data else None,
        )
    # val_dataset = EpisodicDataset(
    #     val_indices,
    #     dataset_dir,
    #     camera_names,
    #     norm_stats,
    #     proprioception=proprioception,
    #     chunk_size=chunk_size,
    #     all_demos=all_demos if preload_data else None,
    #     )
    n_workers = 0
    prefetch = None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=n_workers,
        prefetch_factor=prefetch,
        collate_fn=collate_fn,
        )
    # val_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size_val,
    #     shuffle=True,
    #     pin_memory=True,
    #     num_workers=n_workers,
    #     prefetch_factor=prefetch,
    #     collate_fn=collate_fn,
    #     )
    val_dataloader = None

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
