import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np

pkl_dirs = [
    "/home/ripl/openteach/extracted_data/cups_demonstrations/pickle_files",
    "/home/ripl/openteach/extracted_data/laptop_demonstrations/pickle_files",
    "/home/ripl/openteach/extracted_data/coke_demonstrations/pickle_files",
    "/home/ripl/openteach/extracted_data/blocks_demonstrations2/pickle_files",
]
# stem = '/data3/rlbench_demos/icra_resubmission/converted'
# pkl_dirs = [os.path.join(stem, x) for x in os.listdir(stem)]
for pkl_dir in pkl_dirs:
    demo_lens = []
    for pkl_file in [x for x in os.listdir(pkl_dir) if x.endswith('.pkl')]:
        print(pkl_file)
        pkl_path = os.path.join(pkl_dir, pkl_file)
        with open(pkl_path, 'rb') as f:
            data = pkl.load(f)
        demo_lens.append( len(data['arm_action']))
    print(f"Max demo len: {max(demo_lens)}, Min demo len: {min(demo_lens)}, Mean demo len: {np.mean(demo_lens)}")
    # print the max, min, mean demo lengths on plot somewhere
    plt.hist(demo_lens, bins=30)
    plt.title(pkl_dir)
    plt.xlabel('Demo Length')
    plt.ylabel('Count')
    plt.axvline(x=max(demo_lens), color='r', linestyle='--', label=f'Max: {max(demo_lens)}')
    plt.axvline(x=min(demo_lens), color='g', linestyle='--', label=f'Min: {min(demo_lens)}')
    plt.axvline(x=np.mean(demo_lens), color='b', linestyle='--', label=f'Mean: {np.mean(demo_lens):.2f}')
    plt.legend()
    # save the plot to disk
    plt.savefig(f'{pkl_dir.replace("/", "_")}_demo_lens_hist.png')
    plt.clf()
    plt.close()
    # plt.show()
