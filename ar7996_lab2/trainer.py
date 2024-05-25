from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import time

class Trainer:

    def __init__(self, model):
        self.model = deepcopy(model)
        self._model = model
        self.device = torch.device('cpu')

    def set_trainer_loader(self, train_loader):
        self.train_loader = train_loader

    def set_criterion(self, criterion = "CE"):
        if criterion == "CE":
            self.criterion = nn.CrossEntropyLoss()

    def set_optimizer(self, optimizer:str, lr=0.1, momentum= 0.9, weight_decay= 5e-4):
        self._set_optimizer = optimizer
        if optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer.lower() == "sgdnestrov":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
        elif optimizer.lower() == "adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def set_device(self, device:str):
        if device == "cuda" and not torch.cuda.is_available():
            device = 'cpu'
            print("GPU/CUDA not available. Set to CPU")
        self.device = torch.device(device)    # device = 'cuda', 'cpu'
        self.model.to(self.device)
        self.criterion.to(self.device)

    def train_one_epoch(self):
        epoch_loss = 0
        epoch_acc = 0
        data_time = []
        train_time = []

        self.model.train()

        start_time_data = time.perf_counter()
        for data in self.train_loader:
            end_time_data = time.perf_counter()
            data_time.append(end_time_data - start_time_data)
            x = data[0].to(self.device)
            y = data[1].to(self.device)

            if str(self.device) == "cuda":
                torch.cuda.synchronize()
            start_time_training = time.perf_counter()
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

            start_time_data = time.perf_counter()

        return epoch_loss/len(self.train_loader), epoch_acc/len(self.train_loader), sum(data_time), sum(train_time)


    def train(self, epochs):
        self.epochs = epochs
        self.train_loss = []
        self.train_acc = []

        self.data_load_time = []
        self.train_time = []
        self.running_time = []

        print("Starting Training: ")
        print("Optimizer: {}, num_workers: {}, Device: {}\n".format(self._set_optimizer, self.train_loader.num_workers, str(self.device)))

        for epoch in range(self.epochs):
            start_time = time.perf_counter()
            train_loss, train_acc, data_load_time, train_time = self.train_one_epoch()
            elapsed_time = time.perf_counter() - start_time

            self.data_load_time.append(data_load_time)
            self.train_time.append(train_time)
            self.running_time.append(elapsed_time)

            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)


            print("Epoch: {}/{} Training loss: {}  Training acc: {}".format(epoch+1, self.epochs, train_loss, train_acc))
            print("Data Loading Time: {} secs  Training time: {} secs  Total Running Time: {} secs\n".format(data_load_time, train_time, elapsed_time))
        

    def train_using_pytorch_profiler(self):
        pass

    def accuracy(self, y_pred, y):
        top_pred = y_pred.argmax(1, keepdim =True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc
    
    def reset(self):
        self.model = deepcopy(self._model)
        self.model.to(self.device)
        self.optimizer = None

        self.train_loss = None
        self.train_acc = None

        self.data_load_time = None
        self.train_batch_time = None
        self.running_time = None

    def get_total_data_load_time(self):
        return sum(self.data_load_time)