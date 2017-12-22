import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class hypers():
    def __init__(self):
        self.encoder_hidden_size = 256
        self.decoder_hidden_size = 512
        self.Nz = 128
        self.dropout = 0.9
        self.M = 20

hyper = hypers()

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        # input_size = 5, there are 5 values for each point in the sequence
        self.lstm = nn.LSTM(input_size = 5, hidden_size = hyper.encoder_hidden_size, bias=True, dropout=hyper.dropout, bidirectional=True)
        self.mu = nn.Linear(2 * hyper.encoder_hidden_size, hyper.Nz)
        self.sigma = nn.Linear(2 * hyper.encoder_hidden_size, hyper.Nz)

    def forward(self,inputs,batch_size,hidden_cell=None):
        if hidden_cell is None:
            # Note: hidden state has dims num_layers * num_directions, batch, hidden_size
            # Here, there are 2 directions so thus a single layer
            # batch_size and hidden_size are already defined in params.
            hidden = Variable(torch.zeros(2, batch_size, hyper.encoder_hidden_size).cuda())
            cell =  Variable(torch.zeros(2, batch_size, hyper.encoder_hidden_size).cuda())
            hidden_cell = (hidden, cell)

        # Note: the input size is [131, 100, 5]
        # [Nmax, batch, seq_length]
        # or in torch notation: (seq_len, batch, input_size)

        (hidden,cell) = self.lstm(inputs.float(), hidden_cell)[1]
        # Split hidden in chunks of size = 1 along the first dimension
        # Since the first dimension is 2, it simply grabs each of these values
        # What's stopping using indexing? i.e. hidden_forward = hidden[0,...]
        # Then we don't need squeeze down below
        hidden_forward, hidden_reverse = torch.split(hidden,1,0)
        # size of hidden_forward / hidden_reverse will be [1,batch_size,encoder_hidden_size]
        # squeeze removes all dims of size 1, thus after squeeze they'll both be [batch_size,encoder_hidden_size]
        # concat on the second dimension, i.e. keep batches separate but concatenate hidden features
        hidden_cat = torch.cat([hidden_forward.squeeze(0),hidden_reverse.squeeze(0)],1)
        # Note that hidden_cat is [batch_size,2*encoder_hidden_size]
        mu = self.mu(hidden_cat)
        sigma = self.sigma(hidden_cat)
        # Additionally, z_size is also [batch_size,2*encoder_hidden_size]
        z_size = mu.size()

        # Make normal distributions, which are also [batch_size,2*encoder_hidden_size]
        N = Variable(torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda())

        # Combine mu, sigma and normal
        z = mu + N * torch.exp(sigma/2)
        # Note z has dimensions [batch_size,hyper.Nz] i.e. [100,128]

        return z, mu, sigma

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        # A linear layer takes will take z and create the hidden/cell states
        # The input will be z, i.e. [batch_size,2*encoder_hidden_size] where 2*encoder_hidden_size = Nz
        # The output will be 2*params.decoder_hidden_size - this will be split into hidden/cell states below
        self.hc = nn.Linear(hyper.Nz, 2 * hyper.decoder_hidden_size)

        # Presumably the input_size = params.Nz+5 comes from the fact that the first point is added in,
        # thus the input size is 5 larger than the size of z
        self.lstm = nn.LSTM(input_size = hyper.Nz + 5, hidden_size = hyper.decoder_hidden_size, bias=True, dropout=hyper.dropout)

        # Another fully connected layer projects the hidden state of the LSTM to the output vector y
        # Unlike before, we won't use a non-linear activation here
        # The output is 5*M + M + 3
        # There are M bivariate normal distributions in the Gaussian mixture model that models (delta_x,delta_y)
        # Each bivariate normal distribution contains 5 parameters (hence 5*M)
        # There is another vector of length M which contains the mixture weights of the GMM
        # Finally, there is a categorical distribution (i.e. sums to 1) of size 3 that models the pen state (start line, end line, end drawing)
        # Thus, there are 6*M+3 parameters that need to be modelled for each line in a drawing.
        # Note that M is a hyperparameter.
        self.fc_y = nn.Linear(hyper.decoder_hidden_size, 6 * hyper.M + 3)

        def forward(self, inputs, z, batch_size,hidden_cell=None):
            if hidden_cell is None:
                # Feed z into the linear layer, apply a tanh activation, then split along the second dimension
                # Since the size is 2*params.decoder_hidden_size, splitting is by params.decoder_hidden_size divides it into two parts
                hidden,cell = torch.split(F.tanh(self.hc(z)), hyper.decoder_hidden_size, 1)
                # Now create a tuple, add in an extra dimension in the first position and ensure that it's contiguous in memory
                hidden_cell = (hidden.unsqueeze(0).contiguous(),cell.unsqueeze(0).contiguous())

            # Note input size is [132, 100, 133]
            # This is [Nmax+1, batch, Nmax+1+1]
            # Where the Nmax+1+1 accounts for the fake initial value AND the concatenated z vector

            # Feed everything into the decoder LSTM
            outputs,(hidden,cell) = self.lstm(inputs, hidden_cell)
            # Note: the output size will be [seq_len, batch, hidden_size * num_directions]
            # Thus, [132, batch, hyper.decoder_hidden_size] i.e. [132, 100, 512]

            # There are two modes: training and generate
            # While training, we feed the LSTM with the whole input and use all outputs
            # In generate mode, we just feed the last generated sample

            if self.training:
                # Note: view(-1,...) reshapes the output to a vector of length params.dec_hidden_size
                # i.e. [132, 100, 512] -> [13200, 512]
                y = self.fc_y(outputs.view(-1, hyper.decoder_hidden_size))
            else:
                y = self.fc_y(hidden.view(-1, hyper.decoder_hidden_size))

            # Note y has size [batch*(Nmax+1),hyper.decoder_hidden_size] i.e [13200, 512]
            # Split the output into groups of 6 parameters along the second axis
            # Then stack all but the last one
            # This creates params_mixture of size [M, (Nmax+1)*batch, 6], i.e. the 5 parameters of the bivariate normal distribution and the mixture weight
            # for each of the Nmax lines in each of the batches
            params = torch.split(y,6,1)
            params_mixture = torch.stack(params[:-1])

            # Finally, the last three values are the parameters of the pen at this particular point
            # This has a size [(Nmax+1)*batch, 3]
            params_pen = params[-1]

            # Now split each parameter and label the variables appropriately
            # Each will be of size [M, (Nmax+1)*batch, 1]

            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)

            # Note: Nmax = 131
