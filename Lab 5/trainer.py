from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import time

class Trainer:

    def __init__(self, model):
        self.model = deepcopy(model)
        self._model = model
        self.device = torch.device('cuda')

    def set_trainer_loader(self, train_loader):
        self.train_loader = train_loader
        self.num_batches = len(train_loader)

    def set_criterion(self, criterion = "CE"):
        if criterion == "CE":
            self.criterion = nn.CrossEntropyLoss()

    def set_optimizer(self, optimizer:str, lr=0.1, momentum= 0.9, weight_decay= 5e-4):
        self._optimizer = optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def set_device(self, device:str):
        if "cuda" in device and not torch.cuda.is_available():
            device = 'cpu'
            print("GPU/CUDA not available. Set to CPU")
        self.device = torch.device(device)    # device = 'cuda', 'cpu'
        self.model.to(self.device)
        self.criterion.to(self.device)

    def train_one_epoch(self):
        epoch_loss = 0
        epoch_acc = 0
        train_time = []

        self.model.train()
        start_time_runnning = time.perf_counter()

        for data in self.train_loader:
            start_time_training = time.perf_counter()
            x = data[0].to(self.device)
            y = data[1].to(self.device)

            if str(self.device) == "cuda":
                torch.cuda.synchronize()
            y_pred = self.model(x)
            self.optimizer.zero_grad()
            loss = self.criterion(y_pred, y)
            acc = self.accuracy(y_pred, y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            if str(self.device) == "cuda":
                torch.cuda.synchronize()
            end_time_training = time.perf_counter()
            train_time.append(end_time_training-start_time_training)
        
        end_time_runnning = time.perf_counter()

        return epoch_loss/len(self.train_loader), epoch_acc/len(self.train_loader), sum(train_time), end_time_runnning - start_time_runnning

    def train(self, epochs:int):
        self.epochs = epochs
        self.train_loss = []
        self.train_acc = []
        self.train_time = []
        self.run_time = []

        print("Starting Training: ")
        print("Optimizer: {}, num_workers: {}, Device: {}".format(self._optimizer, self.train_loader.num_workers, str(self.device)))
        print("Number of Devices: {}, Batch Size per GPU: {}".format(1, self.train_loader.batch_size))
        print("Number of Batches: {}\n".format(len(self.train_loader)))

        for epoch in range(self.epochs):
            train_loss, train_acc, train_time, run_time = self.train_one_epoch()
            self.train_time.append(train_time)
            self.run_time.append(run_time)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)
            print("Epoch: {}/{} Training loss: {}  Training acc: {}".format(epoch+1, self.epochs, train_loss, train_acc))
            print("Training time: {} secs, Running time: {} secs\n".format(train_time, run_time))

    def accuracy(self, y_pred, y):
        top_pred = y_pred.argmax(1, keepdim =True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc
    
    def reset(self):
        self.model = deepcopy(self._model)
        self.optimizer = None
        self.train_loss = None
        self.train_acc = None
        self.train_time = None
        self.run_time = None
