
import os
import sys

import torch
# import torchvision
# from torchvision.transforms import v2
from sklearn.metrics import PrecisionRecallDisplay, precision_score, recall_score
from torch import nn
from torchsummary import summary

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from PIL import Image
import tqdm.notebook as tqdm
from torchvision.transforms import v2

# sys.path.append('../dataset_construction')
# from model_training_utils import *



# ---- Plotting ----
plt.figure(figsize=(5, 5));

def plot_loss(log, save=True):
    train_loss = log['train']['loss']
    val_loss = log['val']['loss']
    x = np.arange(1, len(val_loss) + 1)

    if len(train_acc) > 0:
        plt.plot(train_loss, label="training loss")
    plt.plot(val_loss, label="validation loss")

    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.grid()
    plt.title("Cloud-Detection Training and Validation Loss vs Epoch")
    if save:
        plt.savefig("Loss")
        plt.show()
        plt.close()

def plot_accuracy(log, save=True):
    train_acc = log['train']['acc']
    val_acc = log['val']['acc']
    x = np.arange(1, len(val_acc) + 1)

    if len(train_acc) > 0:
        plt.plot(x, train_acc, label="training accuracy")
    plt.plot(x, val_acc, label="validation accuracy")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.title("Cloud-Detection Training and Validation Accuracy vs Epoch")
    plt.grid()
    if save:
        plt.savefig(f"Accuracy")
        plt.show()
        plt.close()


def plot_precision_recall(log, save=True):
    val_precision = log['val']['precision']
    val_recall = log['val']['recall']
    x = np.arange(1, len(val_precision) + 1)
    plt.plot(x, val_precision, label="precision (pos label=1)", )
    plt.plot(x, val_recall, label="recall (pos label=1)")
    plt.legend()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("score")

    plt.title("Validation Precision & Recall vs Epoch")
    if save:
        plt.savefig(f"precision_recall_per_epoch")
        plt.show()
        plt.close()


# def plot_cloudy_mistakes(log, save=True):
#     train_acc = log['train']['cloudy_wrong']
#     val_acc = log['val']['cloudy_wrong']
#
#     plt.plot(train_acc, label="training cloudy_wrong")
#     plt.plot(val_acc, label="validation cloudy_wrong")
#     plt.legend()
#     plt.xlabel("epoch")
#     plt.ylabel("accuracy")
#
#     plt.title("Cloud-Detection Training and percent cloudy misclassifications vs Epoch")
#     if save:
#         plt.savefig(f"cloudy_wrong")
#         plt.close()
#
#
# def plot_clear_mistakes(log, save=True):
#     train_acc = log['train']['clear_wrong']
#     val_acc = log['val']['clear_wrong']
#
#     plt.plot(train_acc, label="training clear_wrong")
#     plt.plot(val_acc, label="validation clear_wrong")
#     plt.legend()
#     plt.xlabel("epoch")
#     plt.ylabel("accuracy")
#
#     plt.title("Cloud-Detection Training and percent clear misclassifications vs Epoch")
#     if save:
#         plt.savefig(f"clear_wrong")
#         plt.close()
#
#
# Utils

def get_device(verbose=False):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"
    if verbose: print(f"Using device {device}")
    return device


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
    if isinstance(m, nn.LazyLinear):
        nn.init.xavier_uniform_(m, gain=nn.init.calculate_gain('relu'))
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m, gain=nn.init.calculate_gain('relu'))
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.xavier_uniform_(m, gain=nn.init.calculate_gain('relu'))
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

class Tester:
    def __init__(self,
                 model,
                 loss_fn,
                 test_loader,
                 img_type='raw-derivative.-60'):
        self.model = model
        self.test_loader = test_loader
        self.img_type = img_type
        self.device = get_device()
        self.loss_fn = loss_fn

    def eval(self):
        ncorrect = 0
        nsamples = 0
        loss_total = 0
        ncloudy_wrong = 0
        nclear_wrong = 0
        preds = torch.tensor([])
        targets = torch.tensor([], dtype=int)

        self.model.eval()
        with torch.no_grad():
            for img_data, y in tqdm.tqdm(self.test_loader, unit="batches"):
                # x = img_data[self.img_type]
                # x = np.stack((img_data[self.img_type], img_data))
                x = img_data
                x = x.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.long)
                scores = self.model(x)
                loss = self.loss_fn(scores, y)
                loss_total += loss.item()

                predictions = torch.argmax(scores, dim=1)
                ncorrect += (predictions == y).sum()
                # print(type(scores), type(y))
                preds = torch.concatenate((preds, torch.amax(scores, dim=1).cpu()))
                targets = torch.concatenate((targets, y.cpu()))

                ncloudy_wrong += ((predictions == 1) & (predictions != y)).cpu().sum()
                nclear_wrong += ((predictions == 0) & (predictions != y)).cpu().sum()
                nsamples += predictions.size(0)
            avg_loss = loss_total / len(self.test_loader)
            acc = float(ncorrect) / nsamples

            report = "{0}: \tloss = {1:.4f},  acc = {2}/{3} ({4:.2f}%)".format(
                'Test', avg_loss, ncorrect, nsamples, acc * 100)

            display = PrecisionRecallDisplay.from_predictions(
                targets.numpy(), preds.numpy(), name="Precision-recall for class 1 (cloudy)", pos_label=1, plot_chance_level=True,
            )
            _ = display.ax_.set_title("2-class Precision-Recall Curve on Test Dataset")
            plt.show()
            plt.close()
            print(report)


class Trainer:

    def __init__(self, model,
                 optimizer,
                 loss_fn,
                 train_loader,
                 val_loader,
                 epochs=1,
                 gamma=0.9,
                 do_summary=True,
                 img_type='raw-derivative.-60',
                 model_save_name='best_cloud_detection_model.pth'
                 ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.gamma = gamma
        self.img_type = img_type
        self.model_save_name = model_save_name

        # self.bprc = BinaryPrecisionRecallCurve(thresholds=None)

        self.cloudy_wrong_data = []
        self.clear_wrong_data = []
        self.device = get_device()
        self.training_log = self.make_training_log()
        if do_summary:
            self.get_model_summary()

    def make_training_log(self):
        training_log = {
            'train': {
                'loss': [],
                'acc': [],
                'cloudy_wrong': [],
                'clear_wrong': []
            },
            'val': {
                'loss': [],
                'acc': [],
                'precision': [],
                'recall': [],
                'cloudy_wrong': [],
                'clear_wrong': []
            }
        }
        return training_log

    def get_model_summary(self):
        """Get the current model configuration."""
        self.model.to(device=self.device)
        self.model.eval()
        with torch.no_grad():
            img_data, y = next(iter(self.train_loader))
            # x = img_data[self.img_type]
            # x = np.stack((img_data['raw-derivative.-60'], img_data['raw-original']))
            # print(img_data.shape)
            x = img_data
            x = x.to(device=self.device, dtype=torch.float)
            self.model(x)
        s = None
        try:
            s = summary(self.model, self.model.input_shape)
            with open('../model_training/model_summary.txt', 'w') as f:
                f.write(str(s))
        except ValueError as verr:
            print(verr)
        return s

    def record_acc_and_loss(self, dataset_type):
        """
        @param dataset_type: 'train' or 'val
        """
        ncorrect = 0
        nsamples = 0
        loss_total = 0
        ncloudy_wrong = 0
        nclear_wrong = 0
        data_loader = self.train_loader if dataset_type == 'train' else self.val_loader
        preds = torch.tensor([])
        targets = torch.tensor([], dtype=int)

        self.model.eval()
        with torch.no_grad():
            for img_data, y in data_loader:
                # x = img_data[self.img_type]
                # x = np.stack((img_data['raw-derivative.-60'], img_data['raw-original']))
                x = img_data
                x = x.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.long)
                scores = self.model(x)
                loss = self.loss_fn(scores, y)
                loss_total += loss.item()

                predictions = torch.argmax(scores, dim=1)
                preds = torch.concatenate((preds, predictions.cpu()))
                targets = torch.concatenate((targets, y.cpu()))

                for i in range(len(predictions)):
                    if predictions[i] == 1 and predictions[i] != y[i]:
                        self.cloudy_wrong_data.append(x[i].cpu())
                    elif predictions[i] == 0 and predictions[i] != y[i]:
                        self.clear_wrong_data.append(x[i].cpu())

                ncorrect += (predictions == y).sum()
                ncloudy_wrong += ((predictions == 1) & (predictions != y)).cpu().sum()
                nclear_wrong += ((predictions == 0) & (predictions != y)).cpu().sum()
                nsamples += predictions.size(0)

            avg_loss = loss_total / len(data_loader)
            acc = float(ncorrect) / nsamples

            self.training_log[dataset_type]['loss'].append(avg_loss)
            self.training_log[dataset_type]['acc'].append(acc)
            self.training_log[dataset_type]['cloudy_wrong'].append(ncloudy_wrong / max(nsamples - float(ncorrect), 1))
            self.training_log[dataset_type]['clear_wrong'].append(nclear_wrong / max(nsamples - float(ncorrect), 1))

            report = "{0}: \tloss = {1:.4f},  acc = {2}/{3} ({4:.2f}%)".format(
                dataset_type.capitalize().rjust(10), avg_loss, ncorrect, nsamples, acc * 100)
            if dataset_type == 'val':
                self.training_log[dataset_type]['precision'].append(
                    precision_score(y_true=targets.numpy(), y_pred=preds.numpy(), average='binary', pos_label=1)
                )
                self.training_log[dataset_type]['recall'].append(
                    recall_score(y_true=targets.numpy(), y_pred=preds.numpy(), average='binary', pos_label=1)
                )
                # display = PrecisionRecallDisplay.from_predictions(
                #     y_true=targets.numpy(), y_pred=preds.numpy(), name="Precision-recall for class 1 (cloudy)", pos_label=1,
                #     plot_chance_level=True,
                # )
                # _ = display.ax_.set_title("2-class Precision-Recall Curve on Validation Dataset")
                # plt.show()
                # plt.close()
            return report

    def train(self, ):
        """
        Train the given model and report accuracy and loss during training.

        Inputs:
        - model: A PyTorch Module giving the model to train.
        - optimizer: An Optimizer object we will use to train the model
        - epochs: (Optional) A Python integer giving the number of epochs to train for

        Returns: dictionary of train and validation loss and accuracy for each epoch.
        """
        # Move model to device
        model = self.model.to(device=self.device)

        # Init LR schedulers
        scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)
        scheduler_plat = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        # scheduler_step = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        try:
            for e in range(1, self.epochs + 1):
                print(f"\n\nEpoch {e}")
                for img_data, y in tqdm.tqdm(self.train_loader, unit="batches"):
                    # Remove the gradients from the previous step
                    self.optimizer.zero_grad()
                    model.train()
                    # x = img_data[self.img_type]
                    x = img_data
                    x = x.to(device=self.device, dtype=torch.float)
                    y = y.to(device=self.device, dtype=torch.long)

                    # Forward pass: compute class scores
                    scores = model(x)
                    loss = self.loss_fn(scores, y)

                    # Backward pass: update weights
                    loss.backward()
                    self.optimizer.step()

                # Update log of train and validation accuracy and loss. Print progress.
                # train_report = self.record_acc_and_loss('train')
                valid_report = self.record_acc_and_loss('val')
                # print(valid_report, '\n', train_report)
                print(valid_report, '\n')

                # Save model parameters with the best validation accuracy
                val_accs = self.training_log['val']['acc']
                if val_accs[-1] == max(val_accs):
                    torch.save(model.state_dict(), f"../model_training/{self.model_save_name}")

                # Update optimizer
                scheduler_exp.step()
                scheduler_plat.step(self.training_log['val']['loss'][-1])
                # scheduler_step.step()
            print('Done training')
            self.make_training_plots(do_save=True)
        except KeyboardInterrupt:
            print('Keyboard Interrupt: Stopping training')
            # self.make_training_plots(do_save=False)

    def make_training_plots(self, do_save):
        plot_accuracy(self.training_log, save=do_save)
        plt.show()
        plt.close()
        plot_loss(self.training_log, save=do_save)
        plt.show()
        plt.close()
        plot_precision_recall(self.training_log, save=do_save)
        plt.show()
        plt.close()
        # plot_cloudy_mistakes(self.training_log, save=do_save)
        # plt.show()
        # plt.close()
        # plot_clear_mistakes(self.training_log, save=do_save)
        # plt.show()
        # plt.close()

# if __name__ == '__main__':
#     from cnn_model import CloudDetection
#     from data_loaders import CloudDetectionTrain
#
#     transform = v2.Compose([
#         v2.ToTensor(),
#         v2.RandomHorizontalFlip(),
#         v2.RandomVerticalFlip(),
#         v2.Normalize(mean=[0], std=[67]),
#         v2.ToDtype(torch.float, scale=True),
#         # v2.ColorJitter(0.5, None, None, None),
#         v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.2, 0.5))
#     ])
#
#     # torchvision.transforms.ToTensor()
#
#     test_prop = 0.0
#     val_prop = 0.3
#
#     labeled_data = CloudDetectionTrain(
#         transform=transform
#     )
#     dataset_size = len(labeled_data)
#     dataset_indices = np.arange(dataset_size)
#
#     np.random.shuffle(dataset_indices)
#
#     # Test / Train split
#     test_split_index = int(np.floor(test_prop * dataset_size))
#     trainset_indices, test_idx = dataset_indices[test_split_index:], dataset_indices[:test_split_index]
#
#     # Train / Val split
#     trainset_size = len(trainset_indices)
#     val_split_index = int(np.floor(val_prop * trainset_size))
#     train_idx, val_idx = trainset_indices[val_split_index:], trainset_indices[:val_split_index]
#
#     batch_size = 64
#
#     # NUM_TRAIN = int(len(labeled_data) * proportion_train)
#     # NUM_TRAIN = NUM_TRAIN - NUM_TRAIN % batch_size
#
#     # val_split_index = int(np.floor(proportion_val * dataset_size))
#     # train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]
#     test_loader = torch.utils.data.DataLoader(
#         dataset=labeled_data,
#         batch_size=batch_size,
#         sampler=torch.utils.data.SubsetRandomSampler(test_idx)
#     )
#
#     train_loader = torch.utils.data.DataLoader(
#         dataset=labeled_data,
#         batch_size=batch_size,
#         sampler=torch.utils.data.SubsetRandomSampler(train_idx)
#     )
#
#     val_loader = torch.utils.data.DataLoader(
#         dataset=labeled_data,
#         batch_size=batch_size,
#         sampler=torch.utils.data.sampler.SubsetRandomSampler(val_idx)
#     )
#
#     img_type = 'raw-derivative.-60'
#
#     learning_rate = 0.001
#     # momentum=0.9
#
#     model = CloudDetection()
#
#     # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6)#, momentum=momentum)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
#     loss_fn = nn.CrossEntropyLoss()
#
#     trainer = Trainer(
#         model, optimizer, loss_fn, train_loader, val_loader,
#         epochs=30, gamma=0.9, do_summary=False,
#         img_type=img_type
#     );
#     trainer.train()