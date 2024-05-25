from data import CIFAR10
from model import Resnet18
from trainer import Trainer
from params import Config


config = Config().parse()
resnet = Resnet18(num_blocks=[2,2,2,2], strides=[1,2,2,2], batch=True)
trainer = Trainer(resnet)
dataset = CIFAR10(config.dataset_path)
trainer.set_criterion("CE")

print("\nC5: GPU vs CPU Training")
print("#######################\n")

#C5: GPU vs CPU Training
#CPU
trainer.set_device('cpu')
trainer.set_trainer_loader(dataset.get_train_loader(batch_size=config.batch_size, num_workers=config.num_workers))
trainer.set_optimizer(optimizer=config.optimizer, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
trainer.train(epochs=config.epochs)
cpu_run_time = trainer.running_time

print("Total Running Time over {} epochs: {}".format(config.epochs, sum(cpu_run_time)))
print("Avg Running Time per epoch: {}\n".format(sum(cpu_run_time)/5))

#GPU
trainer.reset()
trainer.set_device('cuda')
trainer.set_trainer_loader(dataset.get_train_loader(batch_size=config.batch_size, num_workers=config.num_workers))
trainer.set_optimizer(optimizer=config.optimizer, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
trainer.train(epochs=config.epochs)
gpu_run_time = trainer.running_time

print("Total Running Time over {} epochs: {}".format(config.epochs, sum(gpu_run_time)))
print("Avg Running Time per epoch: {}\n".format(sum(gpu_run_time)/5))


print("\nC6:Various Optimizers")
print("#####################\n")
#C6: Various Optimizers
trainer.reset()

opt_list = ["sgd", "adam", "sgdnestrov", "adagrad", "adadelta"]

for opt in opt_list:
    trainer.set_device('cuda')
    trainer.set_trainer_loader(dataset.get_train_loader(batch_size=config.batch_size, num_workers=config.num_workers))
    trainer.set_optimizer(optimizer=opt, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    trainer.train(epochs=config.epochs)
    trainer.reset()
    print("\n")



print("\nC7: Without Batch-Normalization")
print("###############################\n")
#C7: Without Batch-Normalization
resnet = Resnet18(num_blocks=[2,2,2,2], strides=[1,2,2,2], batch=False)
trainer = Trainer(resnet)
trainer.set_criterion("CE")
trainer.set_device('cuda')
trainer.set_trainer_loader(dataset.get_train_loader(batch_size=config.batch_size, num_workers=config.num_workers))
trainer.set_optimizer(optimizer=config.optimizer, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
trainer.train(epochs=config.epochs)