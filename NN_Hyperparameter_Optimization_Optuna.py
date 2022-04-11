import optuna as op
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import import_ipynb
from kafnets import KAF

# path to dataset
data_path = '/domino/datasets/local/NN_Data/'

# features for NN
old_fea_csv = pd.read_csv(data_path + 'feature_list.csv')
old_fea_list = old_fea_csv['Variable Name'].to_list()
old_fea_list_set = set(old_fea_list)
print("Number of features:", len(old_fea_list))

fea_csv = pd.read_csv(data_path + 'new_feature_list.csv')
fea_list = fea_csv['Variable Name'].to_list()
fea_list_set = set(fea_list)
print("Number of features:", len(fea_list))

diff = fea_list_set.difference(old_fea_list_set)
print("Number of replaced features:", len(diff))

# training data
p_in_time_train = pd.read_csv(data_path + 'training_set.csv')

# validation data
p_in_time_valid = pd.read_csv(data_path + 'validation_set.csv')

# mean, std of training data
training_mean, training_std = p_in_time_train[fea_list].mean(), p_in_time_train[fea_list].std()

# Imputation
p_in_time_train[fea_list] = p_in_time_train[fea_list].fillna(training_mean)
p_in_time_valid[fea_list] = p_in_time_valid[fea_list].fillna(training_mean)

# Training and Validation data for NN
X_train, y_train = p_in_time_train[fea_list], 1 - p_in_time_train["label"]
X_valid, y_valid = p_in_time_valid[fea_list], 1 - p_in_time_valid["label"]

# Standardization
X_train_std = (X_train - training_mean)/training_std
X_valid_std = (X_valid - training_mean)/training_std
X_train_std.shape, y_train.shape, X_valid_std.shape, y_valid.shape

from torch.utils.data import Dataset, DataLoader

class NNdataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)
        self.length = self.x.shape[0]
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    
    def __len__(self):
        return self.length

train_dataset = NNdataset(X_train_std.values, y_train.values)
valid_dataset = NNdataset(X_valid_std.values, y_valid.values)

train_dataloader = DataLoader(dataset = train_dataset, shuffle = True, batch_size = 512)
valid_dataloader = DataLoader(dataset = valid_dataset, shuffle = False, batch_size = 512)

from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

class Net(nn.Module):
    def __init__(self, input_shape, dropout_rate, dim1, dim2, act1, act2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, dim1)
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        self.BN1 = nn.BatchNorm1d(dim1)
        
        self.fc2 = nn.Linear(dim1, dim2)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        self.BN2 = nn.BatchNorm1d(dim2)
        
        self.fc3 = nn.Linear(dim2, 1)
        nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('relu'))
        self.dropout = nn.Dropout(dropout_rate)
        
        if act1 == "ReLU":
            self.act1 = nn.ReLU()
        if act1 == "ELU":
            self.act1 = nn.ELU()
        if act1 == "LeakyReLU":
            self.act1 = nn.LeakyReLU()
        if act1 == "ReLU6":
            self.act1 = nn.ReLU6()
            
        if act2 == "ReLU":
            self.act2 = nn.ReLU()
        if act2 == "ELU":
            self.act2 = nn.ELU()
        if act2 == "LeakyReLU":
            self.act2 = nn.LeakyReLU()
        if act2 == "ReLU6":
            self.act2 = nn.ReLU6()
        
    def forward(self, x):
        x = self.dropout(self.act1(self.BN1(self.fc1(x))))
        x = self.dropout(self.act2(self.BN2(self.fc2(x))))
        x = torch.sigmoid(self.fc3(x))
        
        return x

def training(model, optimizer, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    epochs = 200
    patience = 10
    model.to(device)
    
    best_auc, num_patience = 0, 0
    train_loss, valid_loss = [], []
    train_aucs, valid_aucs = [], []
    for epoch in range(epochs):
        train_epoch_loss, valid_epoch_loss = 0, 0
        train_score, train_label = np.zeros(len(y_train)), np.zeros(len(y_train))
        valid_score, valid_label = np.zeros(len(y_valid)), np.zeros(len(y_valid))

        model.train()
        idx = 0
        for i, (train_x, train_y) in enumerate(train_dataloader):
            y_pred = model(train_x.to(device))
            train_score[idx:idx + len(y_pred)] = np.squeeze(y_pred.cpu().detach().numpy(), 1)
            train_label[idx:idx + len(y_pred)] = train_y
            idx += len(y_pred)

            loss = criterion(y_pred, train_y.reshape(-1, 1).to(device))
            train_epoch_loss += loss.cpu().detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        idx = 0
        for i, (valid_x, valid_y) in enumerate(valid_dataloader):
            y_pred = model(valid_x.to(device))
            valid_score[idx:idx + len(y_pred)] = np.squeeze(y_pred.cpu().detach().numpy(), 1)
            valid_label[idx:idx + len(y_pred)] = valid_y
            idx += len(y_pred)

            loss = criterion(y_pred, valid_y.reshape(-1, 1).to(device))
            valid_epoch_loss += loss.cpu().detach().numpy()

        train_auc = roc_auc_score(train_label, train_score)
        valid_auc = roc_auc_score(valid_label, valid_score)

        if (valid_auc < best_auc) and (num_patience < patience): # need patience
            num_patience += 1

        elif (valid_auc < best_auc) and (num_patience == patience): 
            break

        else:
            best_auc = valid_auc
    return best_auc

def objective(trial):
    # integer
    dim1 = trial.suggest_int('dim1', 64, 128, 64)
    dim2 = trial.suggest_int('dim2', 32, 64, 32)
    # float
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log = False)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, log = False)
    # categorical
    act1 = trial.suggest_categorical('act1', ['ReLU', 'ELU', 'LeakyReLU', 'ReLU6'])
    act2 = trial.suggest_categorical('act2', ['ReLU', 'ELU', 'LeakyReLU', 'ReLU6'])

    # Net class is defined outside of objective function
    model = Net(input_shape = len(fea_list), dropout_rate = dropout_rate, dim1 = dim1, dim2 = dim2, act1 = act1, act2 = act2)

    selected_optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    if selected_optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.0)

    criterion  = nn.BCELoss()
    # training function is defined outside of objective function
    best_auc = training(model, optimizer, criterion)

    return best_auc

study = op.create_study(direction = "maximize")
study.optimize(objective, n_trials = 500)
