import h5py
import pickle as pkl
import numpy as np
import os
import argparse
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Convert .pkl files to .h5 format.")
parser.add_argument("folder", type=str, help="Path to the folder containing .pkl files.")
args = parser.parse_args()

def convert_pkl_to_h5(pkl_file):
    # Load the .pkl file
    with open(pkl_file, 'rb') as f:
        data = pkl.load(f)
    # breakpoint()

    h5_file = pkl_file.replace('.pkl', '.h5')
    # breakpoint()

    # Save the data to an .h5 file
    with h5py.File(h5_file, 'w') as h5f:
        # create array of all keys lister
        keys = ['eef_pos', 'eef_quat', 'arm_action', 'gripper_action', 'gripper_state', 'eef_pose', 'joint_pos']
        for key in keys:
            h5f.create_dataset(key, data=data[key])

        # add images
        # breakpoint()
        for i, key in enumerate(['side_cam', 'front_cam', 'overhead_cam']):
            # create a dataset for each key in rgb_frames
            h5f.create_dataset(f'rgb_frames/{key}', data=data['rgb_frames'][:, i])

        # breakpoint()


if __name__ == "__main__":
    # Process the folder
    pkl_folder = args.folder
    pkl_files = [fname for fname in os.listdir(pkl_folder) if fname.endswith('.pkl')]
    for fname in tqdm(pkl_files, desc="Converting files"):
        convert_pkl_to_h5(os.path.join(pkl_folder, fname))
