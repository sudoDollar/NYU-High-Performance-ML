import torchvision.datasets as datasets
from torchvision.transforms import v2
import torch

class CIFAR10:

    def __init__(self, data_path):

        self.transform1 = v2.Compose([
                            v2.ToImage(),
                            v2.RandomCrop(size=(32,32), padding=4),
                            v2.RandomHorizontalFlip(0.5),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
                          ])
        self.transform2 = v2.Compose([
                            v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
                          ])
        
        self.train_set = datasets.CIFAR10(root=data_path, train=True,
                                    download=True, transform=self.transform1)
        
        self.test_set = datasets.CIFAR10(root=data_path, train=False,
                                       download=True, transform=self.transform2)
        

    def get_train_loader(self, batch_size, num_workers=2):
        train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
        
        return train_loader

    def get_test_loader(self, batch_size, num_workers=2):
        test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
        return test_loader