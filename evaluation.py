## pytorch를 깔아야 합니다 ..

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

def make_padding(df):
    scaler = StandardScaler()
    if len(df) == 15:
        list_scaled = scaler.fit_transform(df)
        abc = np.concatenate((np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]),list_scaled,np.array([[0,0,0,0,0,0],[0,0,0,0,0,0]])))
        b = torch.from_numpy(abc)
        instance = [b, 0]
    
    elif len(df) == 19:
        list_scaled = scaler.fit_transform(df)
        abc = np.concatenate((list_scaled, np.array([[0,0,0,0,0,0]])))
        b = torch.from_numpy(abc) 
        instance = [b, 2]

    else:
        list_scaled = scaler.fit_transform(df)
        b = torch.from_numpy(list_scaled)
        instance = [b, 1]
    return instance


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input, hidden=None):
        # lstm step => then ONLY take the sequence's final timetep to pass into the linear/dense layer
        # Note: lstm_out contains outputs for every step of the sequence we are looping over (for BPTT)
        # but we just need the output of the last step of the sequence, aka lstm_out[-1]
        lstm_out, hidden = self.lstm(input, hidden)
        logits = self.linear(lstm_out[-1])              # equivalent to return_sequences=False from Keras
        #softmax = F.softmax(logits)

        return logits, hidden

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = (
            embedding_dim, 2 * embedding_dim
        )
        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
        self.rnn2 = nn.LSTM(
          input_size=self.hidden_dim,
          hidden_size=embedding_dim,
          num_layers=1,
          batch_first=True
        )
    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return  x[:,-1,:]
    
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
          input_size=input_dim,
          hidden_size=input_dim,
          num_layers=1,
          batch_first=True
        )
        self.rnn2 = nn.LSTM(
          input_size=input_dim,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
        self.output_layer = torch.nn.Linear(self.hidden_dim, n_features)
        self.timedist = TimeDistributed(self.output_layer)
        
    def forward(self, x):
        x=x.reshape(-1,1,self.input_dim).repeat(1,self.seq_len,1)       
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        return self.timedist(x)

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)#.to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)#.to(device)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv('data/error_vec.csv', header=None) # 데이터 경로는 바꾸시면 됩니다
data = make_padding(df)
data_loader = DataLoader(data, batch_size = 1)
criterion = nn.L1Loss(reduction='sum')

model1 = LSTM(input_dim = 6, hidden_dim = 3, output_dim = 3, num_layers = 1).to(device)
model1.load_state_dict(torch.load('classification.pt'))
model1.eval()

model2 = RecurrentAutoencoder(20, 6, 64).to(device)
model2.load_state_dict(torch.load('autolstm2.pt'))
model2.eval()

sample = next(iter(data_loader))
sample = sample.float().to(device)

predict, hidden = model1(sample.permute(1,0,2), None)

label = torch.argmax(torch.as_tensor(predict))

predict2 = model2(sample)
score = criterion(predict2, sample)

print(int(label))
print(float(score))