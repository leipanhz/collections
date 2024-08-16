import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import socket

from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(distributed):
    if not distributed:
        return
     
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return

class MyMapDataset(Dataset):
    def __init__(self, data_source):
        super(MyMapDataset, self).__init__()
        self.data_source = list(data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        return item

    def __len__(self):
        return len(self.data_source)


class Trainer:
    def __init__(
        self,
        train_data: DataLoader,
        checkpoint_duration: int,
    ) -> None:
        self.gpu_id = int(os.getenv("LOCAL_RANK") or '0')
        self.train_data = train_data
        self.checkpoint_duration = checkpoint_duration

    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        for batch in self.train_data:
            print(f' {self.gpu_id=}: Running {epoch=}, {batch=}')


    def _save_checkpoint(self, epoch):
        print(f"Epoch {epoch} | Training checkpoint disabled for now.")

    def train(self, max_epochs: int):

        print(f'{max_epochs=}')
        for epoch in range(max_epochs):
            self._run_epoch(epoch)

            if self.gpu_id == 0 and epoch % self.checkpoint_duration == 0:
                self._save_checkpoint(epoch)


def load_dataset(size):
    data = list(range(size))
    dataset = MyMapDataset(data)

    return dataset

def get_params_from_os():
    host = socket.gethostname()
    world_size = int(os.getenv('WORLD_SIZE') or '1')
    rank = int(os.getenv('RANK') or '0')
    print(f'{host=}, {world_size=}, {rank=}')

    return world_size, rank

def build_distributed_sampler(dataset, world_size, rank):
    
    print(f'buld_distributed_sampler: {world_size=}, {rank=}')
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank)
    
    return sampler


def build_data_loader(dataset, sampler, micro_batch_size, num_workers, drop_last, task_collate_fn=None):
    data_loader = DataLoader(dataset,
                            batch_size=micro_batch_size,
                            sampler=sampler,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=drop_last,
                            pin_memory=True,
                            collate_fn=task_collate_fn)
    return data_loader


def eval_dataloader(train_data: DataLoader, epochs: int):
    for epoch in range(epochs):
        train_data.sampler.set_epoch(epoch)
        for batch in train_data:
            print(f' Evaluate samples before Epoch {epoch}, {batch=}')


def main(checkpoint_period: int, total_epochs: int, micro_batch_size: int, dataset_len: int, distributed: bool):
    world_size, rank = get_params_from_os()
    ddp_setup(distributed)
    dataset = load_dataset(dataset_len)
    sampler = build_distributed_sampler(dataset, world_size, rank)
    train_data = build_data_loader(dataset, sampler, micro_batch_size, world_size, True)
    eval_dataloader(train_data, total_epochs)
    trainer = Trainer(train_data, checkpoint_period)
    trainer.train(total_epochs)
    
    if distributed:
        destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--dataset_len', default=8, type=int, help='Total length of the dataset')
    parser.add_argument('--batch_size', default=2, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--checkpoint_period', default=1, type=int, help='How often to save a snapshot')
    parser.add_argument('--distributed', default=False, type=bool, help='Whether training is run locally or in distributed system')
    args = parser.parse_args()
    print(f'{args.checkpoint_period=},  {args.total_epochs=}, {args.dataset_len=}, {args.batch_size=}, {args.distributed=}')

    main(args.checkpoint_period,  args.total_epochs, args.batch_size, args.dataset_len, args.distributed)

