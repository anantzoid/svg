import torch
import torch.nn as nn
from torch.autograd import Variable

class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, bidirectional=False):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=bidirectional, num_layers=self.n_layers)
        if self.bidirectional:
            self.output = nn.Sequential(
                    nn.Linear(2*hidden_size, output_size),
                    nn.Tanh())
        else:
            self.output = nn.Sequential(
                    nn.Linear(hidden_size, output_size),
                    nn.Tanh())
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        if self.bidirectional:
            return (Variable(torch.zeros(self.n_layers*2, self.batch_size, self.hidden_size).cuda()),
                        Variable(torch.zeros(self.n_layers*2, self.batch_size, self.hidden_size).cuda()))
        else:
            return (Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size).cuda()),
                        Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size).cuda()))
        #return hidden

    def forward(self, input):
        if input.dim() == 2:
            input = input.unsqueeze(0)

        embedded = []
        for i in range(input.size()[0]):
            embedded.append(self.embed(input[i].squeeze(0)))
        embedded = torch.stack(embedded)

        h_in = self.lstm(embedded, self.hidden)[0]
        op = torch.stack([self.output(i) for i in h_in])
        if op.size()[0] == 1:
            op = op.squeeze(0)

        return op


class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, bidirectional=False):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=bidirectional, num_layers=self.n_layers)
        if self.bidirectional:
            self.mu_net = nn.Linear(2*hidden_size, output_size)
            self.logvar_net = nn.Linear(2*hidden_size, output_size)
        else:
            self.mu_net = nn.Linear(hidden_size, output_size)
            self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        if self.bidirectional:
            return (Variable(torch.zeros(self.n_layers*2, self.batch_size, self.hidden_size).cuda()),
                        Variable(torch.zeros(self.n_layers*2, self.batch_size, self.hidden_size).cuda()))
        else:
            return (Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size).cuda()),
                        Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size).cuda()))

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        if input.dim() == 2:
            input = input.unsqueeze(0)

        embedded = []
        for i in range(input.size()[0]):
            embedded.append(self.embed(input[i].squeeze(0)))
        embedded = torch.stack(embedded)

        h_in = self.lstm(embedded, self.hidden)[0]
        mu, logvar = [], []
        for i in h_in:
            i = i.squeeze(0)
            mu.append(self.mu_net(i))
            logvar.append(self.logvar_net(i))

        z = torch.stack([self.reparameterize(i, j) for i,j in zip(mu, logvar)])
        mu = torch.stack(mu)
        logvar = torch.stack(logvar)

        if z.size()[0] == 1:
            z = z.squeeze(0)
        return z, mu, logvar
            
