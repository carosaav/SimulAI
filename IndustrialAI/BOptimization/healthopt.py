# This file is part of the
#   SimulAI Project (https://github.com/carosaav/SimulAI).
# Copyright (c) 2020, Perez Colo Ivo, Pirozzo Bernardo Manuel,
# Carolina Saavedra Sueldo
# License: MIT
#   Full Text: https://github.com/carosaav/SimulAI/blob/master/LICENSE


# =====================================================================
# IMPORTS
# =====================================================================

import csv
import numpy as np
import pandas as pd
import attr
from PIL import ImageFile
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import BayesianOptimization
from hyperopt import tpe, hp, fmin, STATUS_OK
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True

np.random.seed(0)
SEED = 123

DATA_SPLIT_PCT = 0.2

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
                "Optimizer: Argument must be a string (adam or sgd)."
            )


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


class Classifier(nn.Module):
    def __init__(self, hidden_size, n_features):
        super(Classifier, self).__init__()
        # lstm
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, batch_first=True
        )
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=4)

    def forward(self, x):
        x, _status = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc1(torch.relu(x))
        x = F.softmax(x, dim=1)
        return x


# ============================================================================
# DATASET
# ============================================================================


class timeseries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


# # ===========================================================================
# # TRAIN & TEST - GAUSSIAN PROCESS
# # ===========================================================================


@attr.s
class BOgp:
    """This class performs Bayesian optimization of
    hyper-parameters using classical Gaussian processing.

    Parameter
    ----------
    name: type
        objective?.
    v_i: list    # [epochs, lr, momentum, bs,
                cv1, cv2, k1, k2, kp1, optim, PATH]
        List of chosen input variables.
    """

    v_i = attr.ib()

    @v_i.validator
    def _validate_v_i(self, attribute, value):
        """Input variables validator."""
        if not isinstance(value, list):
            raise TypeError("v_i: Argument must be a list.")

    def preprocessing(
        self, lr, momentum, batch_size, lookback, hidden_size, opt
    ):
        file_out = pd.read_csv(self.v_i[7].strvalue, low_memory=False)
        file_out = file_out.loc[
            :, ~file_out.columns.astype(str).str.contains("id")
        ]
        file_out = file_out.replace("None", value=np.nan)
        file_out = file_out.dropna()
        file_out = file_out.astype(float, copy=True)

        x = file_out.iloc[:, 1:-1].values
        y = file_out.iloc[:, -1].values
        n_features = x.shape[1]

        def temporalize(X, y, lookback):
            """
            Output
            output_X  A 3D numpy array of shape:
                      ((n_observations-lookback-1) x lookback x
                      n_features)
            output_y  A 1D array of shape:
                      (n_observations-lookback-1), aligned with X.
            """
            output_X = []
            output_y = []
            for i in range(len(X) - lookback - 1):
                t = []
                for j in range(1, lookback + 1):
                    # Gather the past records upto the lookback period
                    t.append(X[[(i + j + 1)], :])
                output_X.append(t)
                output_y.append(y[i + lookback + 1])
            return np.squeeze(np.array(output_X)), np.array(output_y)

        def flatten(X):
            """
            Flatten a 3D array.
            """
            flattened_X = np.empty((X.shape[0], X.shape[2]))
            for i in range(X.shape[0]):
                flattened_X[i] = X[i, (X.shape[1] - 1), :]
            return flattened_X

        def scale(X, scaler):
            """
            Scale 3D array.
            """
            for i in range(X.shape[0]):
                X[i, :, :] = scaler.transform(X[i, :, :])

            return X

        model = Classifier(hidden_size, n_features)
        model.to(device)

        X, Y = temporalize(X=x, y=y, lookback=lookback)
        X_train, X_test, Y_train, Y_test = train_test_split(
            np.array(X),
            np.array(Y),
            test_size=DATA_SPLIT_PCT,
            random_state=SEED,
        )
        X_train = X_train.reshape(X_train.shape[0], lookback, n_features)
        X_test = X_test.reshape(X_test.shape[0], lookback, n_features)

        # Initialize a scaler using the training data
        scaler = StandardScaler().fit(flatten(X_train))
        X_train_scaled = scale(X_train, scaler)
        X_test_scaled = scale(X_test, scaler)

        dataset1 = timeseries(X_train_scaled, Y_train)
        train_loader = DataLoader(dataset1, batch_size)
        dataset2 = timeseries(X_test_scaled, Y_test)
        test_loader = DataLoader(dataset2, batch_size)

        criterion = nn.CrossEntropyLoss()

        if opt == 0:
            opt_method = optim.Adam(model.parameters(), lr=lr)
        else:
            opt_method = optim.SGD(
                model.parameters(), lr=lr, momentum=momentum
            )

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
            "lookback": (self.v_i[4].lower_limit, self.v_i[4].upper_limit),
            "hidden_size": (self.v_i[5].lower_limit, self.v_i[5].upper_limit),
            "opt": (self.v_i[6].lower_limit, self.v_i[6].upper_limit),
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
                "lookback",
                "hidden_size",
                "opt",
            ]
        )

        def process(lr, momentum, batch_size, lookback, hidden_size, opt):
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
                int(lookback),
                int(hidden_size),
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
                    if i % 200 == 199:
                        print(
                            "[%d, %5d] loss: %.3f"
                            % (epoch + 1, i + 1, train_loss / 200)
                        )
                        train_loss = 0.0
                losslist.append(train_loss2)

            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    features, labels = data[0].to(device), data[1].to(device)
                    outputs = model(features)
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
                    int(lookback),
                    int(hidden_size),
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
                "lr", self.v_i[1].lower_limit, self.v_i[1].upper_limit
            ),
            "momentum": hp.uniform(
                "momentum", self.v_i[2].lower_limit, self.v_i[2].upper_limit
            ),
            "batch_size": hp.quniform(
                "batch_size",
                self.v_i[3].lower_limit,
                self.v_i[3].upper_limit,
                1,
            ),
            "lookback": hp.quniform(
                "lookback", self.v_i[4].lower_limit, self.v_i[4].upper_limit, 1
            ),
            "hidden_size": hp.quniform(
                "hidden_size",
                self.v_i[5].lower_limit,
                self.v_i[5].upper_limit,
                1,
            ),
            "opt": hp.quniform(
                "opt", self.v_i[6].lower_limit, self.v_i[6].upper_limit, 1
            ),
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
                "lookback",
                "hidden_size",
                "opt",
            ]
        )

        def process(x):
            lr = x["lr"]
            momentum = x["momentum"]
            batch_size = x["batch_size"]
            lookback = x["lookback"]
            hidden_size = x["hidden_size"]
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
                int(lookback),
                int(hidden_size),
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
                            "[%d, %5d] loss: %.3f"
                            % (epoch + 1, i + 1, train_loss / 20)
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
                    batch_size,
                    lookback,
                    hidden_size,
                    opt,
                ]
            )

            return {
                "loss": -train_loss2,
                "iteration": iteration,
                "acc": acc,
                "lr": lr,
                "momentum": momentum,
                "batch_size": batch_size,
                "lookback": lookback,
                "hidden_size": hidden_size,
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
