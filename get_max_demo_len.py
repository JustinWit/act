import os
import pickle as pkl
from tqdm import tqdm

pkl_dir = "/data3/act_data/real_pick_coke"
max_demo_len = 0
for i in tqdm(range(40)):
    # format like 000, 001, 002, etc
    pkl_path = os.path.join(pkl_dir, f"demo_{i:03d}.pkl")
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    demo_len = data['arm_action'].shape[0]
    max_demo_len = max(max_demo_len, demo_len)

print(max_demo_len)
