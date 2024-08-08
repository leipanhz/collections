import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import os
import socket

import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class MyMapDataset(Dataset):
    def __init__(self, data_source):
        super(MyMapDataset, self).__init__()
        self.data_source = list(data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        return item

    def __len__(self):
        return len(self.data_source)

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        checkpoint_duration: int,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.checkpoint_duration = checkpoint_duration
        self.model = DDP(model, device_ids=[self.gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            print(f'{source=}, {targets=}')
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        # ckp = self.model.module.state_dict()
        # PATH = "checkpoint.pt"
        # torch.save(ckp, PATH)
        # print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
        print(f"Epoch {epoch} | Training checkpoint disabled for now.")

    def train(self, max_epochs: int):

        print(f'{max_epochs=}')
        for epoch in range(max_epochs):
            self._run_epoch(epoch)

            if self.gpu_id == 0 and epoch % self.checkpoint_duration == 0:
                self._save_checkpoint(epoch)

def load_model():
    model = torch.nn.Linear(20, 1)  # sample model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, optimizer

def load_dataset():
    # data = list(range(100))
    # dataset = MyMapDataset(data)
    dataset = MyTrainDataset(16)

    return dataset

def get_params_from_os():
    host = socket.gethostname()
    world_size = int(os.getenv('WORLD_SIZE') or '1')
    rank = int(os.getenv('RANK') or '0')
    print(f'{host=}, {world_size=}, {rank=}')

    return world_size, rank

def build_data_loader(dataset, micro_batch_size, num_workers, drop_last, 
                      world_size, rank, task_collate_fn=None):
    
    print(f'buld_data_loader: {world_size=}, {rank=}')
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank)
    data_loader = DataLoader(dataset,
                            batch_size=micro_batch_size,
                            sampler=sampler,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=drop_last,
                            pin_memory=True,
                            collate_fn=task_collate_fn)
    
    # # Testing with no sampler
    # data_loader = DataLoader(dataset, micro_batch_size, shuffle=True)
    return data_loader

def main(checkpoint_period: int, total_epochs: int, micro_batch_size: int):
    world_size, rank = get_params_from_os()
    ddp_setup()
    model, optimizer = load_model()
    dataset = load_dataset()
    train_data = build_data_loader(dataset, micro_batch_size, world_size - 1, True, world_size, rank)
    trainer = Trainer(model, train_data, optimizer, checkpoint_period)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--checkpoint_period', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    print(f'{args.checkpoint_period=},  {args.total_epochs=}, {args.batch_size=}')

    main(args.checkpoint_period,  args.total_epochs, args.batch_size)

