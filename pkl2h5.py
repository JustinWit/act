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

    h5_file = pkl_file.replace('.pkl', '.h5')

    # Save the data to an .h5 file
    with h5py.File(h5_file, 'w') as h5f:
        # create array of all keys lister
        keys = ['rgb_frames', 'eef_pos', 'eef_quat', 'arm_action', 'gripper_action', 'gripper_state', 'eef_pose']
        for key in keys:
            h5f.create_dataset(key, data=data[key])

        # breakpoint()


if __name__ == "__main__":
    # Process the folder
    pkl_folder = args.folder
    pkl_files = [fname for fname in os.listdir(pkl_folder) if fname.endswith('.pkl')]
    for fname in tqdm(pkl_files, desc="Converting files"):
        convert_pkl_to_h5(os.path.join(pkl_folder, fname))
