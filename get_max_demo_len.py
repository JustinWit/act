import os
import pickle as pkl
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

pkl_dir = "/data3/extracted_data/close_laptop_100/demos"
pkl_dir2 = "datasets/super_slow_arm"
max_demo_len = []
max_demo_len2 = []
for i in tqdm(range(100)):
    # format like 000, 001, 002, etc
    pkl_path = os.path.join(pkl_dir, f"demo_{i:03d}.pkl")
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    max_demo_len.append( len(data['arm_action']))
# for i in tqdm(range(100)):
#     # format like 000, 001, 002, etc
#     pkl_path2 = os.path.join(pkl_dir2, f"demo_{i:03d}.pkl")
#     with open(pkl_path2, 'rb') as f:
#         data2 = pkl.load(f)
#     max_demo_len2.append(data2['arm_action'])

    # demo_len2 = data2['arm_action'].shape[0]
    # max_demo_len = max(max_demo_len, demo_len)
    # max_demo_len2 = max(max_demo_len2, demo_len2)

# mean1 = np.std(np.concatenate(max_demo_len), axis=0)
# mean2 = np.std(np.concatenate(max_demo_len2), axis=0)
breakpoint()
#print(f"mean demo sim: {np.mean(np.concatenate(max_demo_len), axis=0)}")
#print(f"mean demo real: {np.mean(np.concatenate(max_demo_len2), axis=0)}")

    # plt.imshow(data['rgb_frames'][0][0])
    # plt.show()

# print(max_demo_len)
