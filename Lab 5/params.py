import argparse

class Config:

    def __init__(self):
        parser = argparse.ArgumentParser(description='ResNet18 with CIFAR10 profiling')
        parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='input batch size for training (default: 128)')
        parser.add_argument('--num-workers', type=int, default=2,
                            help='Number of I/O processes (default: 2)')
        parser.add_argument('--epochs', type=int, default=2, metavar='N',
                            help='Number of epochs to train (default: 5)')
        parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                            help='learning rate (default: 0.1)')
        parser.add_argument('--device', type=str, default='cuda',
                            help='Device to be used for training(cpu/cuda)')
        parser.add_argument('--optimizer', type=str, default='sgd',
                            help='Optimizer to be used (default: sgd)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='N',
                            help='Momentum Value (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='N',
                            help='Weight Decay value (default 5e-4)')
        parser.add_argument('--dataset-path', type=str, default='./dataset',
                            help='Path of CIFAR10 Dataset (default: ./dataset)')
        
        self._parser = parser

    def parse(self):
        return self._parser.parse_args()
