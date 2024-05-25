from data import CIFAR10
from model import Resnet18
from trainer import Trainer
from params import Config
import matplotlib.pyplot as plt

#C1: Code

#Read parameters from user input
config = Config().parse()

#model
resnet = Resnet18(num_blocks=[2,2,2,2], strides=[1,2,2,2], batch=True)

#helper object to train model
trainer = Trainer(resnet)

#dataset
dataset = CIFAR10(config.dataset_path)
#Create Training set loader
trainer.set_trainer_loader(dataset.get_train_loader(batch_size=config.batch_size, num_workers=config.num_workers))

#Set Optimizer and other parameters as per user input
trainer.set_criterion("CE")
trainer.set_optimizer(optimizer=config.optimizer, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
trainer.set_device(config.device)

#C2: Data loading time, Training Time, Running Time
print("\nC2: Data loading time, Training Time, Running Time")
print("##################################################\n")

#After each epoch, following method will print required times.

#Train Model
trainer.train(epochs=config.epochs)


print("\nC3: I/O Optimization")
print("####################\n")
#C3: I/O Optimization
trainer.reset()
num_workers = 0
data_time = float('inf')
data_time_list = []
num_workers_list = []

while True:
    num_workers_list.append(num_workers)
    trainer.set_trainer_loader(dataset.get_train_loader(batch_size=config.batch_size, num_workers=num_workers))
    trainer.set_optimizer(optimizer=config.optimizer, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    trainer.train(epochs=config.epochs)
    print("Total Data Time: ", trainer.get_total_data_load_time(), "\n")
    if trainer.get_total_data_load_time() <= data_time:
        data_time_list.append(trainer.get_total_data_load_time())
        num_workers += 4
        data_time = data_time_list[-1]
        trainer.reset()
    else:
        data_time_list.append(trainer.get_total_data_load_time())
        break

num_workers -= 4

print("Optimal Num Workers: ", num_workers)
file_path = "optimal_num_workers.txt"
with open(file_path, "w") as file:
    file.write(str(num_workers))


#Plot the graph
fig = plt.figure()
plt.plot(num_workers_list, data_time_list)
plt.scatter(num_workers_list, data_time_list)
# plt.text(4, 5.129909634590149, "3.13")
# plt.text(8, 6.057909976691008, "4.06")
plt.xlabel("Num Workers")
plt.ylabel("Data Loading Time (secs)")
plt.title("Data Loading Time vs Num Workers over 5 epochs")
plt.savefig('c3.png')

print("\nC4: Profiling with Optimal Number of Workers")
print("############################################\n")
#C4: Profiling with above num_workers value
trainer.reset()
trainer.set_trainer_loader(dataset.get_train_loader(batch_size=config.batch_size, num_workers=1))
trainer.set_optimizer(optimizer=config.optimizer, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
trainer.train(epochs=config.epochs)
worker_1_data_time = trainer.data_load_time
worker_1_train_time = trainer.train_time
print("Total Data Loading time using {} workers: {}".format(1, sum(worker_1_data_time)))
print("Total Computing time using {} workers: {}\n".format(1, sum(worker_1_train_time)))


trainer.reset()
trainer.set_trainer_loader(dataset.get_train_loader(batch_size=config.batch_size, num_workers=num_workers))
trainer.set_optimizer(optimizer=config.optimizer, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
trainer.train(epochs=config.epochs)
worker_n_data_time = trainer.data_load_time
worker_n_train_time = trainer.train_time
print("Total Data Loading time using {} workers: {}".format(num_workers, sum(worker_n_data_time)))
print("Total Computing time using {} workers: {}\n".format(num_workers, sum(worker_n_train_time)))


#Plot the graph
epochs = []
for ep in range(config.epochs):
    epochs.append(ep+1)

fig = plt.figure()
plt.plot(epochs, worker_1_data_time, label = "Num Workers: 1")
plt.plot(epochs, worker_n_data_time, color='r', label = "Num Workers: {}".format(num_workers))
plt.scatter(epochs, worker_1_data_time)
plt.scatter(epochs, worker_n_data_time, color='r')
plt.xlabel("Epochs")
plt.ylabel("Data Loading Time (secs)")
plt.title("Data Loading Time per epcoh")
plt.legend()
plt.savefig('c4_data.png')

fig = plt.figure()
plt.plot(epochs, worker_1_train_time, label = "Num Workers: 1")
plt.plot(epochs, worker_n_train_time, color='r', label = "Num Workers: {}".format(num_workers))
plt.scatter(epochs, worker_1_train_time)
plt.scatter(epochs, worker_n_train_time, color='r')
plt.xlabel("Epcohs")
plt.ylabel("Training Time (secs)")
plt.title("Training Time per epoch")
plt.legend()
plt.savefig('c4_compute.png')

print("Graphs comparing Data Loading time and Computing Time has been saved.\n")











