import torch
from data import CIFAR10
from model import Resnet18
from trainer import Trainer
from ParallelTrainer import ParallelTrainer
from params import Config


config = Config().parse()
resnet = Resnet18(num_blocks=[2,2,2,2], strides=[1,2,2,2], batch=True)
trainer = Trainer(resnet)
dataset = CIFAR10(config.dataset_path)
trainer.set_criterion("CE")

#Single GPU
batch_size = 32
results_gpu_1 = []
while True:
    print(batch_size)
    trainer.set_device(config.device)
    trainer.set_trainer_loader(dataset.get_train_loader(batch_size=batch_size, num_workers=config.num_workers))
    trainer.set_optimizer(optimizer=config.optimizer, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    try:
        trainer.train(epochs=config.epochs)
        #Tuple of Batch Size, Train Time, Run Time, Num of Batches/Iterations per GPU
        res = (batch_size, trainer.train_time[1], trainer.run_time[1], trainer.num_batches)
        results_gpu_1.append(res)

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory error occurred. Please reduce batch size\n")
        else:
            print("A RuntimeError occurred:", e)
        break
    
    batch_size *= 4
    trainer.reset()

#2 GPUs
batch_size_per_gpu = 32
results_gpu_2 = []
trainer = ParallelTrainer(resnet, 2)
trainer.set_criterion("CE")
while True:
    print(batch_size_per_gpu)
    trainer.set_device(config.device)
    trainer.set_trainer_loader(dataset.get_train_loader(batch_size=batch_size_per_gpu*2, num_workers=config.num_workers))
    trainer.set_optimizer(optimizer=config.optimizer, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    try:
        trainer.train(epochs=config.epochs)
        res = (batch_size_per_gpu, trainer.train_time[1], trainer.run_time[1], trainer.num_batches)
        results_gpu_2.append(res)

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory error occurred. Please reduce batch size\n")
        else:
            print("A RuntimeError occurred:", e)
        break
    
    batch_size_per_gpu *= 4
    trainer.reset()


#4 GPUs
batch_size_per_gpu = 32
results_gpu_4 = []
trainer = ParallelTrainer(resnet, 4)
trainer.set_criterion("CE")
while True:
    print(batch_size_per_gpu)
    trainer.set_device(config.device)
    trainer.set_trainer_loader(dataset.get_train_loader(batch_size=batch_size_per_gpu*4, num_workers=config.num_workers))
    trainer.set_optimizer(optimizer=config.optimizer, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    try:
        trainer.train(epochs=config.epochs)
        res = (batch_size_per_gpu, trainer.train_time[1], trainer.run_time[1], trainer.num_batches)
        results_gpu_4.append(res)

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory error occurred. Please reduce batch size\n")
        else:
            print("A RuntimeError occurred:", e)
        break
    
    batch_size_per_gpu *= 4
    trainer.reset()

#Q2 SpeedUp Measurement
l = len(results_gpu_1)
for i in range(l):
    print("Batch Size per GPU: {}\n".format(results_gpu_1[i][0]))
    print("Training Time for 1 GPU: {}\n".format(results_gpu_1[i][1]))

    print("Training Time for 2 GPUs: {}".format(results_gpu_2[i][1]))
    print("SpeedUp of 2 GPU over 1: {}\n".format(results_gpu_1[i][2]/results_gpu_2[i][2]))

    print("Training Time for 4 GPUs: {}".format(results_gpu_4[i][1]))
    print("SpeedUp of 4 GPU over 1: {}\n\n".format(results_gpu_1[i][2]/results_gpu_4[i][2]))


#Q3.1 Computation vs Communication
l = len(results_gpu_1)
for i in range(l):
    print("Batch Size per GPU: {}\n".format(results_gpu_1[i][0]))
    compute_time_per_image = results_gpu_1[i][1]/50000
    compute_time_per_batch = compute_time_per_image * results_gpu_1[i][0]

    #T(Compute, 2) = T(Compute, Batch, 1) * (#Batch – 1) + T(Compute, Image, 1) * (#images in last batch)
    compute_time = compute_time_per_batch * (results_gpu_2[i][3] - 1) + compute_time_per_image*(25000 - (results_gpu_2[i][3]-1)*results_gpu_1[i][0])
    print("Compute Time for 2 GPUs: {}".format(compute_time))
    print("Communication Time for 2 GPUs: {}\n".format(results_gpu_2[i][1] - compute_time))

    compute_time = compute_time_per_batch * (results_gpu_4[i][3] - 1) + compute_time_per_image*(12500 - (results_gpu_4[i][3]-1)*results_gpu_1[i][0])
    print("Compute Time for 4 GPUs: {}".format(compute_time))
    print("Communication Time for 4 GPUs: {}\n\n".format(results_gpu_4[i][1] - compute_time))

#Q3.2 Communication bandwidth utilization
l = len(results_gpu_1)
num_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)

for i in range(l):
    print("Batch Size per GPU: {}".format(results_gpu_1[i][0]))
    compute_time_per_image = results_gpu_1[i][1]/50000
    compute_time_per_batch = compute_time_per_image * results_gpu_1[i][0]

    params_size_2 = results_gpu_2[i][3] * num_params * 4 / 1000000000 #in GB
    compute_time = compute_time_per_batch * (results_gpu_2[i][3] - 1) + compute_time_per_image*(25000 - (results_gpu_2[i][3]-1)*results_gpu_1[i][0])
    print("Time for AllReduce in 2 GPUs setup: {} secs".format(results_gpu_2[i][1] - compute_time))
    print("Bandwidth Utilization in 2 GPUs setup: {} GB/s\n".format(2*params_size_2*(2 - 1)/(results_gpu_2[i][1] - compute_time)))

    params_size_4 = results_gpu_4[i][3] * num_params * 4 / 1000000000 #in GB
    compute_time = compute_time_per_batch * (results_gpu_4[i][3] - 1) + compute_time_per_image*(12500 - (results_gpu_4[i][3]-1)*results_gpu_1[i][0])
    print("Time for AllReduce in 4 GPUs setup: {} secs".format(results_gpu_4[i][1] - compute_time))
    print("Bandwidth Utilization in 4 GPUs setup: {} GB/s\n\n".format(2*params_size_4*(4 - 1)/(results_gpu_4[i][1] - compute_time)))


#Q4
trainer = Trainer(resnet)
trainer.set_criterion("CE")
trainer.set_device(config.device)
trainer.set_trainer_loader(dataset.get_train_loader(batch_size=128, num_workers=config.num_workers))
trainer.set_optimizer(optimizer=config.optimizer, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
trainer.train(epochs=5)


batch_size_per_gpu //= 4
trainer = ParallelTrainer(resnet, 4)
trainer.set_criterion("CE")
trainer.set_device(config.device)
trainer.set_trainer_loader(dataset.get_train_loader(batch_size=batch_size_per_gpu*4, num_workers=config.num_workers))
trainer.set_optimizer(optimizer=config.optimizer, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
trainer.train(epochs=5)