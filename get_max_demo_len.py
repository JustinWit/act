import os
import pickle as pkl
from tqdm import tqdm

import matplotlib.pyplot as plt

pkl_dir = "/data3/act_data/sim_pick_coke_500_domain_rand"
max_demo_len = 0
for i in tqdm(range(20)):
    # format like 000, 001, 002, etc
    pkl_path = os.path.join(pkl_dir, f"demo_{i:03d}.pkl")
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    demo_len = data['arm_action'].shape[0]
    max_demo_len = max(max_demo_len, demo_len)

    # plt.imshow(data['rgb_frames'][0][0])
    # plt.show()

print(max_demo_len)
