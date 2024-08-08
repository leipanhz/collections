import torch
from torch.utils.data import IterableDataset, DataLoader


class MyIterableDataset(IterableDataset):
    # def __init__(self, data_source_func, *args, **kwargs):
    #     super(MyIterableDataset, self).__init__()
    #     self.data_source_func = data_source_func
    #     self.args = args
    #     self.kwargs = kwargs

    # def __iter__(self):
    #     # Create the generator inside the iter method
    #     generator = self.data_source_func(*self.args, **self.kwargs)
    #     for item in generator:
    #         yield item
    
    def __init__(self, data_source):
        super(MyIterableDataset, self).__init__()
        self.data_source = data_source  

    def __iter__(self):
        # This method should yield data samples one at a time
        for item in self.data_source:
            # Perform any necessary preprocessing or transformations
            yield item

    # Optional: Implement __len__ if the length is known
    def __len__(self):
        return len(self.data_source)


def data_generator():
    for i in range(1000):  # Simulate a large dataset
        print(f'data generator: yield {i}')
        yield i

def build_data_loader(dataset, micro_batch_size, num_workers, drop_last,
        task_collate_fn=None):

    # Sampler.
    # world_size = 2  #mpu.get_data_parallel_world_size()
    # rank =[0,1] #mpu.get_data_parallel_rank()
    # sampler = torch.utils.data.distributed.DistributedSampler(
    #     dataset, num_replicas=world_size, rank=rank)      
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset)

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=micro_batch_size,
                                              sampler=sampler,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last=drop_last,
                                              pin_memory=True,
                                              collate_fn=task_collate_fn)

    return data_loader


if __name__ == "__main__":
    # data = [1, 2, 3, 4, 5]  
    data = data_generator()
    dataset = MyIterableDataset(data)
    # dataloader = DataLoader(dataset, batch_size=None, num_workers=4)
    dataloader = build_data_loader(dataset, 2, 4, True)

    print("print data")
    for data in dataloader:
        print(data)


