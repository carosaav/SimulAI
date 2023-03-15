# This file is part of the
#   SimulAI Project (https://github.com/carosaav/SimulAI).
# Copyright (c) 2020, Perez Colo Ivo, Pirozzo Bernardo Manuel,
# Carolina Saavedra Sueldo
# License: MIT
#   Full Text: https://github.com/carosaav/SimulAI/blob/master/LICENSE


# =====================================================================
# IMPORTS
# =====================================================================

import os
import csv
import fnmatch
import numpy as np
import pandas as pd
import attr
from PIL import ImageFile
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import BayesianOptimization
from hyperopt import tpe, hp, fmin, Trials, STATUS_OK
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# ============================================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True

np.random.seed(0)
torch.manual_seed(0)

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We're using =>", device)

# ============================================================================
# INPUT-OUTPUT VARIABLES
# ============================================================================


@attr.s
class DiscreteVariable:
    """This class validates the discrete variables that make up the
       search space, the lower and upper limits defined, as
       well as the type of data entered.

    Parameters
    ----------
    name: str
        Name of the variable.
    lower_limit: positive int
        Lower limit of the variable. Should be a positive integer.
    upper_limit: positive int
        Upper limit of the variable. Should be a positive integer.
    """

    name = attr.ib()
    lower_limit = attr.ib()
    upper_limit = attr.ib()

    @name.validator
    def _validate_name(self, attribute, value):
        """Name validator.
        Parameters
        ----------
        value: str
            User-selected value.
        """
        if not isinstance(value, str):
            raise TypeError("Name: Argument must be a string.")

    @lower_limit.validator
    def _validate_lower_limit(self, attribute, value):
        """Lower limit validator.
        Parameters
        ----------
        value: int
            User-selected value.
        """
        if not isinstance(value, int):
            raise TypeError("Lower Limit: Argument must be an integer.")
        if value < 0:
            raise ValueError("Lower limit: Argument must be higher than 0.")

    @upper_limit.validator
    def _validate_upper_limit(self, attribute, value):
        """Upper limit validator.
        Parameters
        ----------
        value: int
            User-selected value.
        """
        if not isinstance(value, int):
            raise TypeError("Upper Limit: Argument must be an integer.")
        if value < 0:
            raise ValueError("Upper Limit: Argument must be higher than 0.")


@attr.s
class ContinuousVariable:
    """This class validates the continuous variables that make up the
       search space, the lower and upper limits defined, as
       well as the type of data entered.

    Parameters
    ----------
    name: str
        Name of the variable.
    lower_limit: positive int
        Lower limit of the variable. Should be a positive integer.
    upper_limit: positive int
        Upper limit of the variable. Should be a positive integer.
    """

    name = attr.ib()
    lower_limit = attr.ib()
    upper_limit = attr.ib()

    @name.validator
    def _validate_name(self, attribute, value):
        """Name validator.
        Parameters
        ----------
        value: str
            User-selected value.
        """
        if not isinstance(value, str):
            raise TypeError("Name: Argument must be a string.")

    @lower_limit.validator
    def _validate_lower_limit(self, attribute, value):
        """Lower limit validator.
        Parameters
        ----------
        value: float
            User-selected value.
        """
        if not isinstance(value, float):
            raise TypeError("Lower Limit: Argument must be an float.")
        if value < 0:
            raise ValueError("Lower limit: Argument must be higher than 0.")

    @upper_limit.validator
    def _validate_upper_limit(self, attribute, value):
        """Upper limit validator.
        Parameters
        ----------
        value: float
            User-selected value.
        """
        if not isinstance(value, float):
            raise TypeError("Upper Limit: Argument must be an float.")
        if value < 0:
            raise ValueError("Upper Limit: Argument must be higher than 0.")


@attr.s
class StringVariable:
    """This class validates the string variables that make up the
       search space and the type of data entered.

    Parameters
    ----------
    name: str
        Name of the variable.
    strvalue: str
        Selected optimizer
    """

    name = attr.ib()
    strvalue = attr.ib()

    @name.validator
    def _validate_name(self, attribute, value):
        """Name validator.
        Parameters
        ----------
        value: str
            User-selected value.
        """
        if not isinstance(value, str):
            raise TypeError("Name: Argument must be a string.")

    @strvalue.validator
    def _strvalue(self, attribute, value):
        """Strvalue validator.
        Parameters
        ----------
        strvalue: float
            User-selected value.
        """
        if not isinstance(value, str):
            raise TypeError(
                "Optimizer: Argument must be a string (adam or sgd).")


@attr.s
class NumEpochs:
    """This class validates the epochs number.

    Parameter
    ----------
    quantity: int
        Number of epochs.
    """

    quantity = attr.ib()

    @quantity.validator
    def _quantity(self, attribute, value):
        """Quantity validator.
        Parameters
        ----------
        value: int
            User-selected value.
        """
        if not isinstance(value, int):
            raise TypeError("Epochs number: Argument must be an integer.")


# # ===========================================================================
# # NEURAL NETWORK
# # ===========================================================================


class Binary_Classifier(nn.Module):
    def __init__(self, cv1, cv2, k1, k2, kp1, device):
        super(Binary_Classifier, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=cv1, kernel_size=k1)
        # max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=kp1, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=cv1, out_channels=cv2, kernel_size=k2)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 16)
        self.fc4 = nn.Linear(16, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        dimension = list(x.shape)
        x = x.view(-1, dimension[1] * dimension[2] * dimension[3])
        if device == "cuda":
            fc1 = nn.Linear(
                in_features=dimension[1] * dimension[2] * dimension[3],
                out_features=256
            ).to(torch.device("cuda:0"))
        else:
            fc1 = nn.Linear(
                in_features=dimension[1] * dimension[2] * dimension[3],
                out_features=256
            )
        x = F.relu(fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.softmax(F.relu(self.fc4(x)))
        return x


# # ===========================================================================
# # TRAIN & TEST - GAUSSIAN PROCESS
# # ===========================================================================


@attr.s
class BOgp:
    """This class performs Bayesian optimization of hyper-parameters
    using classical Gaussian processing.

    Parameter
    ----------
    name: type
        objective?.
    v_i: list    # [epochs, lr, momentum, bs, cv1,
                    cv2, k1, k2, kp1, optim, PATH]
        List of chosen input variables.
    """

    v_i = attr.ib()

    @v_i.validator
    def _validate_v_i(self, attribute, value):
        """Input variables validator."""
        if not isinstance(value, list):
            raise TypeError("v_i: Argument must be a list.")

    def preprocessing(
                   self, lr, momentum, batch_size, cv1, cv2, k1, k2, kp1, opt):
        model = Binary_Classifier(cv1, cv2, k1, k2, kp1, device)
        model.to(device)

        t_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        data = datasets.ImageFolder(
            self.v_i[10].strvalue, transform=t_transform)
        total_data_size = len(
            fnmatch.filter(os.listdir(
                self.v_i[10].strvalue + "/defective"), "*.*")
        ) + len(
            fnmatch.filter(os.listdir(
                self.v_i[10].strvalue + "/non-defective"), "*.*")
        )
        train_data_size = int(total_data_size * 0.8)
        test_data_size = total_data_size - train_data_size
        train_data, test_data = random_split(
            data, [train_data_size, test_data_size])
        train_loader = DataLoader(train_data, batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()

        if opt == 0:
            opt_method = optim.Adam(model.parameters(), lr=lr)
        else:
            opt_method = optim.SGD(
                model.parameters(), lr=lr, momentum=momentum)

        return train_loader, test_loader, model, opt_method, device, criterion

    # # =======================================================================
    # # BAYESIAN OPTIMIZATION
    # # =======================================================================

    def boptimization(self):
        losslist = []
        y_true = []
        y_pred = []
        space = {
            "lr": (self.v_i[1].lower_limit, self.v_i[1].upper_limit),
            "momentum": (self.v_i[2].lower_limit, self.v_i[2].upper_limit),
            "batch_size": (self.v_i[3].lower_limit, self.v_i[3].upper_limit),
            "cv1": (self.v_i[4].lower_limit, self.v_i[4].upper_limit),
            "cv2": (self.v_i[5].lower_limit, self.v_i[5].upper_limit),
            "k1": (self.v_i[6].lower_limit, self.v_i[6].upper_limit),
            "k2": (self.v_i[7].lower_limit, self.v_i[7].upper_limit),
            "kp1": (self.v_i[8].lower_limit, self.v_i[8].upper_limit),
            "opt": (self.v_i[9].lower_limit, self.v_i[9].upper_limit),
        }

        # File to save csv results
        out_file = "./results.csv"
        of_connection = open(out_file, "w")
        writer = csv.writer(of_connection)
        writer.writerow(
            [
                "iteration",
                "loss",
                "accuracy",
                "lr",
                "momentum",
                "batch_size",
                "cv1",
                "cv2",
                "k1",
                "k2",
                "kp1",
                "opt",
            ]
        )

        def process(lr, momentum, batch_size, cv1, cv2, k1, k2, kp1, opt):
            (
                train_loader,
                test_loader,
                model,
                opt_method,
                device,
                criterion,
            ) = self.preprocessing(
                lr,
                momentum,
                int(batch_size),
                int(cv1),
                int(cv2),
                int(k1),
                int(k2),
                int(kp1),
                int(opt),
            )
            total = 0
            correct = 0
            for epoch in range(self.v_i[0].quantity):
                train_loss = 0.0
                train_loss2 = 0.0
                model.train()
                for i, data in enumerate(train_loader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    opt_method.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    opt_method.step()
                    train_loss += loss.item()
                    train_loss2 += loss.item()
                    if i % self.v_i[0].quantity == (self.v_i[0].quantity - 1):
                        print(
                            "[%d, %5d] loss: %.3f" % (
                                epoch + 1, i + 1, train_loss / 20)
                        )
                        train_loss = 0.0
                losslist.append(train_loss2)

            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    y_true.append(labels.cpu().numpy())
                    y_pred.append(predicted.cpu().numpy())

            acc = 100 * correct / total
            global iteration
            iteration += 1
            writer.writerow(
                [
                    iteration,
                    train_loss2,
                    acc,
                    lr,
                    momentum,
                    int(batch_size),
                    int(cv1),
                    int(cv2),
                    int(k1),
                    int(k2),
                    int(kp1),
                    int(opt),
                ]
            )

            return acc

        optimizer = BayesianOptimization(process, space, verbose=2)

        # If we want to set a known starting point
        # optimizer.probe(
        #     params={"x": 0.5, "y": 0.7},
        #     lazy=True,
        # )

        # New optimizer is loaded with previously seen points
        # load_logs(new_optimizer, logs=["./logs.json"]);

        global iteration
        iteration = 0
        optimizer.maximize(
            init_points=2,
            n_iter=100,
            acq="ei",
        )

        print("Best Parameter Setting: {}".format(optimizer.max["params"]))
        print("Best Target Value: {}".format(optimizer.max["target"]))

        for i, res in enumerate(optimizer.res):
            print("Iteration {}: \n\t{}".format(i, res))

        logger = JSONLogger(path="./logs.json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    __boptimization = boptimization


# # ===========================================================================
# # TRAIN & TEST TREE PARZEN ESTIMATOR
# # ===========================================================================


@attr.s
class BOtpe(BOgp):
    """This class performs Bayesian optimization of
    hyper-parameters using Tree-Parzen Estimator.

    Parameter
    ----------
    v_i: list    # [epochs, lr, momentum, bs, cv1, cv2,
                    k1, k2, kp1, optim, PATH]
        List of chosen input variables.
    """

    # # =======================================================================
    # # BAYESIAN OPTIMIZATION
    # # =======================================================================

    def boptimization(self):
        losslist = []
        y_true = []
        y_pred = []
        space = {
            "lr": hp.uniform(
                "lr", self.v_i[1].lower_limit, self.v_i[1].upper_limit),
            "momentum": hp.uniform(
                "momentum", self.v_i[2].lower_limit, self.v_i[2].upper_limit
            ),
            "batch_size": hp.quniform(
                "batch_size", self.v_i[3].lower_limit,
                self.v_i[3].upper_limit, 1
            ),
            "cv1": hp.quniform(
                "cv1", self.v_i[4].lower_limit, self.v_i[4].upper_limit, 1
            ),
            "cv2": hp.quniform(
                "cv2", self.v_i[5].lower_limit, self.v_i[5].upper_limit, 1
            ),
            "k1": hp.quniform(
                "k1", self.v_i[6].lower_limit, self.v_i[6].upper_limit, 1
            ),
            "k2": hp.quniform(
                "k2", self.v_i[7].lower_limit, self.v_i[7].upper_limit, 1
            ),
            "kp1": hp.quniform(
                "kp1", self.v_i[8].lower_limit, self.v_i[8].upper_limit, 1
            ),
            "opt": hp.quniform(
                "opt", self.v_i[9].lower_limit, self.v_i[9].upper_limit, 1
            ),
        }

        # Initialize trials object
        trials = Trials()
        # File to save csv results
        out_file = "./results.csv"
        of_connection = open(out_file, "w")
        writer = csv.writer(of_connection)
        writer.writerow(
            [
                "iteration",
                "loss",
                "accuracy",
                "lr",
                "momentum",
                "batch_size",
                "cv1",
                "cv2",
                "k1",
                "k2",
                "kp1",
                "opt",
            ]
        )

        def process(x):
            lr = x["lr"]
            momentum = x["momentum"]
            batch_size = x["batch_size"]
            cv1 = x["cv1"]
            cv2 = x["cv2"]
            k1 = x["k1"]
            k2 = x["k2"]
            kp1 = x["kp1"]
            opt = x["opt"]
            correct = 0
            total = 0

            (
                train_loader,
                test_loader,
                model,
                opt_method,
                device,
                criterion,
            ) = self.preprocessing(
                lr,
                momentum,
                int(batch_size),
                int(cv1),
                int(cv2),
                int(k1),
                int(k2),
                int(kp1),
                int(opt),
            )
            for epoch in range(self.v_i[0].quantity):
                train_loss = 0.0
                train_loss2 = 0.0
                model.train()
                for i, data in enumerate(train_loader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    opt_method.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    opt_method.step()
                    train_loss += loss.item()
                    train_loss2 += loss.item()
                    if i % self.v_i[0].quantity == (self.v_i[0].quantity - 1):
                        print(
                            "[%d, %5d] loss: %.3f" % (
                                epoch + 1, i + 1, train_loss / 20)
                        )
                        train_loss = 0.0
                losslist.append(train_loss2)

            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    y_true.append(labels.cpu().numpy())
                    y_pred.append(predicted.cpu().numpy())

            acc = 100 * correct / total
            global iteration
            iteration += 1
            writer.writerow(
                [
                    iteration,
                    losslist,
                    acc,
                    lr,
                    momentum,
                    batch_size,
                    cv1,
                    cv2,
                    k1,
                    k2,
                    kp1,
                    opt,
                ]
            )

            return {
                "loss": -losslist,
                "iteration": iteration,
                "acc": acc,
                "lr": lr,
                "momentum": momentum,
                "batch_size": batch_size,
                "cv1": cv1,
                "cv2": cv2,
                "k1": k1,
                "k2": k2,
                "kp1": kp1,
                "opt": opt,
                "status": STATUS_OK,
            }

        global iteration
        iteration = 0
        optimizer = fmin(
            fn=process,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
        )

        print("Best: {}".format(optimizer))

        # Dataframe of results from optimization
        tpe_results = pd.DataFrame(
            {
                "loss": [x["loss"] for x in trials.results],
                "iteration": [x["iteration"] for x in trials.results],
                "acc": [x["acc"] for x in trials.results],
                "lr": [x["lr"] for x in trials.results],
                "momentum": [x["momentum"] for x in trials.results],
                "batch_size": [x["batch_size"] for x in trials.results],
                "cv1": [x["cv1"] for x in trials.results],
                "cv2": [x["cv2"] for x in trials.results],
                "k1": [x["k1"] for x in trials.results],
                "k2": [x["k2"] for x in trials.results],
                "kp1": [x["kp1"] for x in trials.results],
                "opt": [x["opt"] for x in trials.results],
            }
        )
        tpe_results.to_csv("results_tpe.csv")
