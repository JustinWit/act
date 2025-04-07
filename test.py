import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from itertools import cycle


dataset1 = torch.tensor([0, 1, 2, 3, 4, 5])
dataset2 = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 23, 123, 25, 34, 56, 78, 89])


dataloader1 = DataLoader(dataset1, batch_size=2, shuffle=True, num_workers=1)
dataloader2 = DataLoader(dataset2, batch_size=3, shuffle=True, num_workers=1)

small_inter = iter(dataloader1)

# breakpoint()

for epoch in range(2):
    print("Epoch: ", epoch)
    for i, data1 in enumerate(dataloader2):
        try:
            data2 = next(small_inter)
        except StopIteration:
            small_inter = iter(dataloader1)
            data2 = next(small_inter)

        print(data1, data2)

# (aloha) ripl@ripl-d3:~/act$ python test.py
# Epoch:  0
# tensor([ 23, 123,  89]) tensor([2, 1])
# tensor([15, 10, 11]) tensor([4, 3])
# tensor([16, 25, 13]) tensor([5, 0])
# tensor([34, 78, 14]) tensor([1, 0])
# tensor([17, 12, 56]) tensor([4, 3])
# Epoch:  1
# tensor([10, 78, 11]) tensor([5, 2])
# tensor([12, 23, 56]) tensor([4, 5])
# tensor([ 15,  13, 123]) tensor([3, 0])
# tensor([17, 25, 89]) tensor([1, 2])
# tensor([34, 14, 16]) tensor([1, 5])