import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import os
import socket


class MyMapDataset(Dataset):
    def __init__(self, data_source):
        super(MyMapDataset, self).__init__()
        self.data_source = list(data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        return item

    def __len__(self):
        return len(self.data_source)


def build_data_loader(dataset, micro_batch_size, num_workers, drop_last,
        task_collate_fn=None):

    # Create sampler based on rank, world_size and dataset
    host = socket.gethostname()
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    print(f'{host=}, {world_size=}, {rank=}')
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=micro_batch_size,
                                              sampler=sampler,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              drop_last=drop_last,
                                              pin_memory=True,
                                              collate_fn=task_collate_fn)

    return data_loader


if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    dataset = MyMapDataset(data)

    # TODO: read arugment in the settings.xml
    dataloader = build_data_loader(dataset, 2, 0, True)

    print("print data")
    for batch in dataloader:
        print(batch)


