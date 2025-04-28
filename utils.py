import numpy as np
import torch
import os
import h5py
import pickle as pkl
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from deoxys_transform_utils import quat2axisangle, axisangle2quat, quat_multiply, mat2quat
import h5py
from torchvision.transforms import v2
# from scipy.spatial.transform import Rotation as R

import IPython
e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self,
            episode_ids,
            dataset_dir,
            camera_names,
            norm_stats,
            chunk_size=None,
            all_demos=None,
            preload_to_cpu=False,
            preload_to_gpu=False,
            crop_resize=False,
            ):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.preload_to_gpu = preload_to_gpu
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.crop_resize = crop_resize
        # convert to tensor and move to gpu
        for k, v in self.norm_stats.items():
            if isinstance(v, np.ndarray):
                self.norm_stats[k] = torch.from_numpy(v).float()
                if preload_to_gpu:
                    self.norm_stats[k] = self.norm_stats[k].cuda()
        self.is_sim = None
        # self.__getitem__(0) # initialize self.is_sim
        self.is_sim = False
        self.chunk_size = chunk_size
        self.all_demos = all_demos

        self.img_transform = v2.Compose([
            v2.RandomResizedCrop(size=(720, 1280) if self.norm_stats['full_size_img'] else (360, 640), scale=(0.8, 1.0)),
        ])


    def set_norm_stats(self, norm_stats):
        self.norm_stats = norm_stats
        # convert to tensor and move to gpu
        for k, v in self.norm_stats.items():
            if isinstance(v, np.ndarray):
                self.norm_stats[k] = torch.from_numpy(v).float()
                if self.preload_to_gpu:
                    self.norm_stats[k] = self.norm_stats[k].cuda()

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        """
        OpenVLA image augs
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        """
        sample_full_episode = False # hardcode

        ##############################
        ### Training from h5 files ###
        if self.all_demos is None:
            # use f-string to format index to be three digits with leading zeros
            episode_id = f'{self.episode_ids[index]:03d}'
            dataset_path = os.path.join(self.dataset_dir, f'demo_{episode_id}.h5')
            root = h5py.File(dataset_path, 'r')

            # get random timestep
            episode_len = root['arm_action'].shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            action_len = episode_len - start_ts

            # get image at start_ts
            imgs = []
            for cam in self.camera_names:
                cam_img = root[f'rgb_frames/{cam}'][start_ts: start_ts + 1]
                imgs.append(cam_img)
            cam_image = np.concatenate(imgs, axis=0)
            cam_image = preproc_imgs(np.expand_dims(cam_image, axis=0), full_size_img=self.norm_stats['full_size_img'])
            if self.crop_resize:
                cam_image = self.img_transform(cam_image)

            # get proprioception at start_ts
            if self.norm_stats['use_proprioception']:
                qpos = get_single_proprioception(
                    root['gripper_state'][start_ts],
                    root['eef_pose'][start_ts],
                    root['eef_pos'][start_ts],
                    gripper_proprio=self.norm_stats['use_gripper_proprio']
                )
                qpos = torch.from_numpy(qpos).to(device='cuda' if self.preload_to_gpu else "cpu", dtype=torch.float32)
            else:
                qpos = torch.zeros(self.norm_stats['qpos_mean'].shape, device='cuda' if self.preload_to_gpu else "cpu")

            # get action chunk
            action = get_action_chunk(
                root['eef_pos'][start_ts:],
                root['eef_quat'][start_ts:],
                root['arm_action'][start_ts:],
                root['gripper_action'][start_ts:],
                absolute=self.norm_stats['absolute_actions']
            )
            padded_action = torch.zeros((episode_len, action.shape[1]), device='cuda' if self.preload_to_gpu else "cpu")
            padded_action[:action_len] = action
            is_pad = torch.zeros(episode_len, device='cuda' if self.preload_to_gpu else "cpu", dtype=torch.bool)
            is_pad[action_len:] = 1

        ####################################
        ### Training from preloaded data ###
        # TODO: set this up for rocoda data
        else:
            # get data at index
            data = self.all_demos[index]

            # get random timestep
            episode_len = data['action'].shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            action_len = episode_len - start_ts

            # get image at start_ts
            cam_image = data['camera'][start_ts: start_ts + 1]

            # get proprioception at start_ts
            if self.norm_stats['use_proprioception']:
                qpos = data['qpos'][start_ts].to(device='cuda' if self.preload_to_gpu else "cpu")
            else:
                qpos = torch.zeros(self.norm_stats['qpos_mean'].shape, device='cuda' if self.preload_to_gpu else "cpu")

            # get action chunk
            action = data['action'][start_ts:]
            padded_action = torch.zeros_like(data['action'])
            padded_action[:action_len] = action
            is_pad = torch.zeros(episode_len, device='cuda' if self.preload_to_gpu else "cpu", dtype=torch.bool)
            is_pad[action_len:] = 1


        # normalize data
        padded_action = (padded_action - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        # added this since the actions just get truncated by the model anyways
        padded_action = padded_action[:self.chunk_size]
        is_pad = is_pad[:self.chunk_size]
        return cam_image, qpos, padded_action, is_pad


def get_norm_stats(
    dataset_dir,
    num_episodes,
    proprioception=True,
    preload_to_cpu=False,
    preload_to_gpu=False,
    gripper_proprio=False,
    absolute_actions=False,
    full_size_img=False,
    ):
    all_qpos_data = []
    all_action_data = []
    all_demos = []
    total_n = 0
    listdir = [x for x in os.listdir(dataset_dir) if x.endswith('.h5')]
    for i, episode_path in enumerate(tqdm(listdir, desc='Loading data')):
        # dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        # with open(os.path.join(dataset_dir, episode_path), 'rb') as dbfile:
        root = h5py.File(os.path.join(dataset_dir, episode_path), 'r')
        qpos = get_proprioception(root, gripper_proprio=gripper_proprio)
        if not proprioception:
            qpos = np.zeros_like(qpos)
        action = get_action(root, absolute=absolute_actions)
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        if preload_to_cpu or preload_to_gpu:
            all_demos.append({
                "qpos": torch.from_numpy(qpos).float(),
                "action": torch.from_numpy(action).float(),
                "camera": preproc_imgs(root['rgb_frames'], full_size_img=full_size_img),
            })
        if preload_to_gpu:
            for k, v in all_demos[-1].items():
                all_demos[-1][k] = v.cuda().contiguous()
        total_n += len(action)

    all_qpos_data = torch.vstack(all_qpos_data)
    # if not proprioception:
    #     all_qpos_data = torch.zeros_like(all_qpos_data)
    all_action_data = torch.vstack(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = all_action_data.std(dim=[0], keepdim=True)
    # action_std = torch.clip(action_std, 1e-2, np.inf) # clipping # NOTE: moved to imitate_episode so we clip after combining datasets

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
    # qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping # NOTE: moved to imitate_episode so we clip after combining datasets

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": None,  # qpos,  # maybe delete this ?
        "use_proprioception": proprioception,
        'use_gripper_proprio': gripper_proprio,
        "absolute_actions": absolute_actions,
        "full_size_img": full_size_img,
        "total_n": total_n,
        }
    return stats, all_demos if len(all_demos) > 0 else None


def get_action(root, absolute=False):
    if absolute:
        pos = root['eef_pos'][:].squeeze() + root['arm_action'][:, :3]
        quat_rot_actions = [axisangle2quat(x) for x in root['arm_action'][:, 3:]]
        rot = np.array([
            quat2axisangle(quat_multiply(i, j)) if quat2axisangle(quat_multiply(i, j))[0] > 0.0 else quat2axisangle(-quat_multiply(i, j)) for i,j in \
                zip(quat_rot_actions, root['eef_quat'])
                ])
        arm_action = np.hstack((pos, rot))
    else:
        arm_action = root['arm_action']
    action = np.hstack((arm_action, np.expand_dims(root['gripper_action'][:], axis=1)))
    return action


def get_action_chunk(eef_pos, eef_quat, arm_action, gripper_action, absolute=False):
    if absolute:
        pos = eef_pos.squeeze() + arm_action[:, :3]
        quat_rot_actions = [axisangle2quat(x) for x in arm_action[:, 3:]]
        rot = np.array([
            quat2axisangle(quat_multiply(i, j)) if quat2axisangle(quat_multiply(i, j))[0] > 0.0 else quat2axisangle(-quat_multiply(i, j)) for i,j in \
                zip(quat_rot_actions, eef_quat)
                ])

        arm_action = np.hstack((pos, rot))

    action = np.hstack((arm_action, np.expand_dims(gripper_action, axis=1)))
    action = torch.from_numpy(action).float()
    return action


def preproc_imgs(imgs, full_size_img=False):
    # TODO: Handle mid-def or downsize to 360
    if imgs.shape[1] == 3:  # real data has 3 cams
        # imgs = imgs[:, 2]  # we only use the front cam
        imgs = imgs.squeeze()

        # imgs = imgs[:, :, 280: 1000]
        # # assert imgs.shape[1:] == (720, 720, 3)
        # if not full_size_img:
        #     # downsize to 256 x 256
        #     resized_imgs = []
        #     for i in range(imgs.shape[0]):
        #         resized_imgs.append(cv2.resize(imgs[i], (256, 256)))
        #     imgs = np.stack(resized_imgs, axis=0)
    # elif imgs.shape[1] == 1:  # sim data has one cam
    #     imgs = imgs[:, 0]
    else:
        raise ValueError('Unknown camera shape')
    if full_size_img:
        assert imgs.shape[1:] == (720, 1280, 3)
    else:
        # downsize to 640 x 360
        imgs = np.stack([cv2.resize(imgs[i], (640, 360)) for i in range(imgs.shape[0])])
        assert imgs.shape[1:] == (360, 640, 3)
    # convert bgr to rgb
    imgs = imgs[..., ::-1]
    imgs = torch.from_numpy(imgs.copy()).float() / 255.0
    imgs = torch.einsum('k h w c -> k c h w', imgs)
    return imgs


def get_proprioception(data, gripper_proprio=False):
    gripper = np.expand_dims(data['gripper_state'][:], 1)
    if np.isclose(gripper.max(), 0.04, atol=1e-2):
        gripper *= 2
    # NOTE 'eef_quat' is "current_quat" which is negated based on "target_quat". We want to use raw robot quats
    raw_quats = [mat2quat(x[:3, :3]) for x in data['eef_pose']]
    axis_angle = np.concatenate([[quat2axisangle(i)] if quat2axisangle(i)[0] > 0.0 else [quat2axisangle(-i)] for i in raw_quats])

    if not gripper_proprio:
        gripper = np.zeros_like(gripper)
    eef_pos = data['eef_pos'][:]
    qpos = np.hstack((eef_pos.squeeze(), axis_angle, gripper))
    return qpos

def get_single_proprioception(gripper, eef_pose, eef_pos, gripper_proprio=False):
    raw_quat = mat2quat(eef_pose[:3, :3])
    axis_angle = quat2axisangle(raw_quat) if quat2axisangle(raw_quat)[0] > 0.0 else quat2axisangle(-raw_quat)
    if not gripper_proprio:
        gripper = np.zeros_like(gripper)
    qpos = np.hstack((eef_pos.squeeze(), axis_angle, gripper))
    return qpos

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
    preload_to_cpu=False,
    preload_to_gpu=False,
    gripper_proprio=False,
    absolute_actions=False,
    full_size_img=False,
    crop_resize=False,
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
        preload_to_cpu=preload_to_cpu,
        preload_to_gpu=preload_to_gpu,
        gripper_proprio=gripper_proprio,
        absolute_actions=absolute_actions,
        full_size_img=full_size_img,
        )

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        train_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        chunk_size=chunk_size,
        all_demos=all_demos,
        preload_to_cpu=preload_to_cpu,
        preload_to_gpu=preload_to_gpu,
        crop_resize=crop_resize,
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
        pin_memory=not preload_to_gpu,
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

def combined_std(mean1, std1, n1, mean2, std2, n2, combined_mean):
    numerator = (n1 * std1**2 + n2 * std2**2 + n1 * (mean1 - combined_mean)**2 + n2 * (mean2 - combined_mean)**2)
    denominator = n1 + n2
    return np.sqrt(numerator / denominator)