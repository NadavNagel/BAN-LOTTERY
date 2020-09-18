# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from tensorboardX import SummaryWriter
from ban import config
from ban.updater import BANUpdater
from common.logger import Logger
from ban.models.pruning import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.01) ##0.01 mnist, 0.001 cifar
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--resume_gen", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--print_interval", type=int, default=50)
    parser.add_argument("--decay", type=float, default=1)
    parser.add_argument("--experiment_name", type=str, default='test')
    parser.add_argument("--prune", type=str, default=None)
    parser.add_argument("--init_teq", type=str, default='random')
    parser.add_argument("--n_epoch", type=int, default=10)
    parser.add_argument("--n_gen", type=int, default=10)
    parser.add_argument("--label", type=str, default='soft')
    parser.add_argument("--window", type=int, default=0)

    args = parser.parse_args()
    window = args.window
    label = args.label
    n_gen = args.n_gen
    n_epoch = args.n_epoch
    init_teq = args.init_teq
    prune = args.prune
    experiment_name = args.experiment_name
    results_path = os.path.join(args.outdir, experiment_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    logger = Logger(args)
    logger.print_args(n_epoch, n_gen)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == "cifar10":
        trainset = CIFAR10(root="./data",
                           train=True,
                           download=True,
                           transform=transform)
        testset = CIFAR10(root="./data",
                          train=False,
                          download=True,
                          transform=transform)
    else:
        trainset = MNIST(root="./data",
                         train=True,
                         download=True,
                         transform=transform)
        testset = MNIST(root="./data",
                        train=False,
                        download=True,
                        transform=transform)
        train_size = int(0.8 * len(trainset))
        test_size = len(trainset) - train_size
        trainset, valset = torch.utils.data.random_split(trainset, [train_size, test_size])

    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=True)
    validation_loader = DataLoader(valset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(testset,
                             batch_size=args.batch_size,
                             shuffle=False)

    model = config.get_model(args.dataset, results_path).to(device)

    if args.weight:
        model.load_state_dict(torch.load(args.weight))
    mask, step = make_mask(model)  ## for pruning
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kwargs = {
        "model": model,
        "optimizer": optimizer,
        "n_gen": n_gen,
    }

    writer = SummaryWriter()
    updater = BANUpdater(**kwargs)
    criterion = nn.CrossEntropyLoss()

    i = 0
    last_acc = 0
    print("train...")
    for gen in range(args.resume_gen, n_gen):
        if prune == 'begining_before' and gen == 0:
            mask, step = prune_by_percentile(90, mask, model)
        temp_lr = args.lr
        train_loss_list = []
        val_loss_list = []
        iterations = []
        acc_test_list = []
        acc_val_list = []
        gen_models_path = os.path.join(results_path, "gen_" + str(gen))
        if not os.path.exists(gen_models_path):
            os.makedirs(gen_models_path)
        for epoch in range(n_epoch):
            train_loss = 0
            train_counter = 0
            for idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                t_loss = updater.update(inputs, targets, criterion, prune, window, label).item() #######
                train_loss += t_loss
                train_counter += 1
                i += 1
                if i % args.print_interval == 0:
                    writer.add_scalar("train_loss", t_loss, i)
                    val_loss = 0
                    correct = 0
                    total = 0
                    with torch.no_grad():
                    ## Validation
                        for idx, (inputs, targets) in enumerate(validation_loader):
                            inputs, targets = inputs.to(device), targets.to(device)
                            outputs = updater.model(inputs)
                            loss = criterion(outputs, targets).item()
                            val_loss += loss
                            val_loss += loss
                            _, pred = outputs.max(1)
                            total += targets.size(0)
                            correct += pred.eq(targets).sum().item()
                    val_loss /= len(test_loader)
                    acc = 100. * correct / total
                    acc_val_list.append(acc)


                    if acc >= last_acc:
                        last_model_weight = os.path.join(results_path, args.dataset + "_gen_" + str(gen) + ".pth.tar")
                        torch.save(updater.model.state_dict(), last_model_weight)
                        last_acc = acc
                    else:
                        temp_lr = args.decay*temp_lr
                        optimizer = optim.Adam(model.parameters(), lr=temp_lr)
                        print('updated learing rate:', temp_lr)
                        updater.optimizer = optimizer
                    writer.add_scalar("val_loss", val_loss, i)

                    ## Test
                    test_loss_ = 0
                    correct_test = 0
                    total_test = 0
                    with torch.no_grad():
                        for idx, (inputs_test, targets_test) in enumerate(test_loader):
                            inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
                            outputs_test = updater.model(inputs_test)
                            loss_test = criterion(outputs_test, targets_test).item()
                            test_loss_ += loss_test
                            test_loss_ += loss_test
                            _, pred_test = outputs_test.max(1)
                            total_test += targets_test.size(0)
                            correct_test += pred_test.eq(targets_test).sum().item()
                    test_loss_ /= len(test_loader)
                    acc_test = 100. * correct_test / total_test
                    acc_test_list.append(acc_test)



                    if train_counter <= args.print_interval:
                        nun_interval = train_counter
                    else:
                        nun_interval = args.print_interval
                    logger.print_log(epoch, i, train_loss / nun_interval, val_loss, acc)
                    train_loss_list.append(train_loss / nun_interval)
                    val_loss_list.append(val_loss)
                    iterations.append(i)
                    train_loss = 0
                    train_counter = 0
            last_model_weight_temp = os.path.join(gen_models_path, args.dataset + "_gen_" + str(gen) +"_epoch_" + str(epoch) + ".pth.tar")
            torch.save(updater.model.state_dict(), last_model_weight_temp)
        updater.register_last_model(last_model_weight, args.dataset)
        updater.gen += 1
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        total_non_zero_param = sum(p.nonzero().size(0) for p in model.parameters())
        print('old model param:', pytorch_total_params, 'non-zero', total_non_zero_param)
        print('save losses, gen:', gen)
        np.savez(os.path.join(results_path, args.dataset + "_gen_" + str(gen) + ".loss.npz"),
                 train_loss_list=np.array(train_loss_list),
                 val_loss_list=np.array(val_loss_list),
                 iterations=np.array(iterations),
                 total_params=np.array(pytorch_total_params),
                 total_non_zero_param=np.array(total_non_zero_param),
                 acc_test_list=np.array(acc_test_list),
                 acc_val_list=np.array(acc_val_list)

        )
        if prune == 'every_gen':
            mask, step = prune_by_percentile((gen + 1) * 10, mask, model, old_mask=None, prune_net=False)
        # if prune == 'begining_after' and gen == 0:
        #     mask, step = prune_by_percentile(80, mask, model)
        print("Born Again...")
        last_acc = 0
        if init_teq == 'fix':
            model = config.get_model(args.dataset, results_path, gen=0, epoch=0).to(device)
        elif init_teq == 'increase':
            model = config.get_model(args.dataset, results_path, gen=max(gen - 1, 0), epoch=max(gen - 1, 0) ).to(device)
        elif init_teq == 'decrease':
            model = config.get_model(args.dataset, results_path, gen=max(gen - 1, 0), epoch= 9 - max(gen - 1, 0)).to(device)
        elif init_teq == 'random':
            model = config.get_model(args.dataset, results_path, gen=None, epoch=None).to(device)
        if prune == 'every_gen':## for pruning
            old_mask = mask
            mask, step = prune_by_percentile((gen + 1) * 10, mask, model, old_mask=old_mask)
        if prune == 'begining_after' and gen == 0:
            mask, step = prune_by_percentile(90, mask, model)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        total_non_zero_param = sum(p.nonzero().size(0) for p in model.parameters())
        print('new model param:', pytorch_total_params, 'non-zero', total_non_zero_param)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        updater.model = model
        updater.optimizer = optimizer



if __name__ == "__main__":
    main()

    # main(experiment_name='tester', prune='begining_after', init_teq='random'  , n_epoch=10,n_gen=2  ,label='hard',window=0)
    # main(experiment_name='3.0.1', prune='begining_before', init_teq='random' , n_epoch=10,n_gen=1  ,label='soft',window=0)
    # main(experiment_name='3.1.1', prune='begining_after', init_teq='random'  , n_epoch=10,n_gen=2  ,label='soft',window=0)
    # main(experiment_name='3.1.2', prune='begining_after', init_teq='fix'     , n_epoch=10,n_gen=2  ,label='soft',window=0)
    # main(experiment_name='3.2.1', prune='every_gen'     , init_teq='random'  , n_epoch=10,n_gen=10 ,label='soft',window=0)
    # main(experiment_name='3.2.2', prune='every_gen'     , init_teq='fix'     , n_epoch=10,n_gen=10 ,label='soft',window=0)
    # main(experiment_name='3.2.3', prune='every_gen'     , init_teq='increase', n_epoch=10,n_gen=10 ,label='soft',window=0)
    # main(experiment_name='3.2.4', prune='every_gen'     , init_teq='decrease', n_epoch=10,n_gen=10 ,label='soft',window=0)
    # main(experiment_name='1.1'  , prune='none'          , init_teq='random'  , n_epoch=10,n_gen=10 ,label='soft',window=0)
    # main(experiment_name='1.2.1', prune='none'          , init_teq='fix'     , n_epoch=10,n_gen=10 ,label='soft',window=0)
    # main(experiment_name='1.2.2', prune='none'          , init_teq='increase', n_epoch=10,n_gen=10 ,label='soft',window=0)
    # main(experiment_name='1.2.3', prune='none'          , init_teq='decrease', n_epoch=10,n_gen=10 ,label='soft',window=0)
    # main(experiment_name='3.1.1_h', prune='begining_after', init_teq='random'  , n_epoch=10,n_gen=2  ,label='hard',window=0)
    # main(experiment_name='3.1.2_h', prune='begining_after', init_teq='fix'     , n_epoch=10,n_gen=2  ,label='hard',window=0)
    # main(experiment_name='1.1_h'  , prune='none'          , init_teq='random'  , n_epoch=10,n_gen=10 ,label='hard',window=0)
    # main(experiment_name='1.2.1_h', prune='none'          , init_teq='fix'     , n_epoch=10,n_gen=10 ,label='hard',window=0)
    # main(experiment_name='1.2.2_h', prune='none'          , init_teq='increase', n_epoch=10,n_gen=10 ,label='hard',window=0)
    # main(experiment_name='1.2.3_h', prune='none'          , init_teq='decrease', n_epoch=10,n_gen=10 ,label='hard',window=0)
    # main(experiment_name='3.2.1_h', prune='every_gen'     , init_teq='random'  , n_epoch=10,n_gen=10 ,label='hard',window=0)
    # main(experiment_name='3.2.2_h', prune='every_gen'     , init_teq='fix'     , n_epoch=10,n_gen=10 ,label='hard',window=0)
    # main(experiment_name='3.2.3_h', prune='every_gen'     , init_teq='increase', n_epoch=10,n_gen=10 ,label='hard',window=0)
    # main(experiment_name='3.2.4_h', prune='every_gen'     , init_teq='decrease', n_epoch=10,n_gen=10 ,label='hard',window=0)
    # main(experiment_name='4.2.1', prune='every_gen'     , init_teq='random'  , n_epoch=10,n_gen=10 ,label='hard',window=2)
    # main(experiment_name='4.2.2', prune='every_gen'     , init_teq='fix'     , n_epoch=10,n_gen=10 ,label='hard',window=2)
    # main(experiment_name='4.2.3', prune='every_gen'     , init_teq='increase', n_epoch=10,n_gen=10 ,label='hard',window=2)
    # main(experiment_name='4.2.4', prune='every_gen'     , init_teq='decrease', n_epoch=10,n_gen=10 ,label='hard',window=2)
    # main(experiment_name='5.2.1', prune='every_gen'     , init_teq='random'  , n_epoch=10,n_gen=10 ,label='hard',window=3)
    # main(experiment_name='5.2.2', prune='every_gen'     , init_teq='fix'     , n_epoch=10,n_gen=10 ,label='hard',window=3)
    # main(experiment_name='5.2.3', prune='every_gen'     , init_teq='increase', n_epoch=10,n_gen=10 ,label='hard',window=3)
    # main(experiment_name='5.2.4', prune='every_gen'     , init_teq='decrease', n_epoch=10,n_gen=10 ,label='hard',window=3)