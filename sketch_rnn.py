# import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from custom_dataloader import *

# todo data augmentation: see supplementary section 3
# todo encoder = 512, decoder = 2048

class hypers():
    def __init__(self):
        self.encoder_hidden_size = 256
        self.decoder_hidden_size = 512
        self.Nz = 128
        self.dropout = 0.9
        self.M = 20
        self.max_seq_length = 200
        self.lr = 0.001
        self.eta_min = 0.01
        self.data = 'octopus.npz'
        self.batch_size = 100
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.5
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 1.

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

            # Note: text implies that hidden state is used in training, whilst
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
            if self.training:
                len_out = sketches.Nmax + 1
            else:
                len_out = 1

            pi = F.softmax(pi.transpose(0,1).squeeze()).view(len_out,-1,hyper.M)
            mu_x = mu_x.transpose(0,1).squeeze().contiguous().view(len_out,-1,hyper.M)
            mu_y = mu_y.transpose(0,1).squeeze().contiguous().view(len_out,-1,hyper.M)
            sigma_x = torch.exp(sigma_x.transpose(0,1).squeeze()).view(len_out,-1,hyper.M)
            sigma_y = torch.exp(sigma_y.transpose(0,1).squeeze()).view(len_out,-1,hyper.M)
            rho_xy = torch.tanh(rho_xy.transpose(0,1).squeeze()).view(len_out,-1,hyper.M)
            q = F.softmax(params_pen).view(len_out,-1,3)

            return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell

class Model():
    def __init__(self):
        self.encoder = encoder().cuda()
        self.decoder = decoder().cuda()
        self.encoder_optim = optim.Adam(self.encoder.parameters(),hyper.lr)
        self.decoder_optim = optim.Adam(self.decoder.paramters(),hyper.lr)
        self.eta_step = hyper.eta_min

    def train(self, epoch):
        self.encoder.train()
        self.decoder.train()

        batch, lengths = sketches.get_batch(hyper.batch_size)

        z, self.mu, self.sigma = self.encoder(batch, hyper.batch_size)

        sos = Variable(torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * hyper.batch_size).cuda()).unsqueeze(0)
        batch_init = torch.cat([sos, batch], 0)
        z_stack = torch.stack([z] * (sketches.Nmax + 1))
        inputs = torch.cat([batch_init, z_stack], 2)

        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, self.q, _, _ = self.decoder(inputs, z)

        mask, dx, dy, p = sketches.get_target(batch,lengths)

        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()

        self.eta_step = 1 - (1 - hyper.eta_min) * hyper.R

        LKL = self.kullback_leibler_loss()
        LR = self.reconstruction_loss(mask, dx, dy, p, epoch)
        loss = LR + LKL

        loss.backward()

        nn.utils.clip_grad_norm(self.encoder.parameters(), hyper.grad_clip)
        nn.utils.clip_grad_norm(self.decoder.parameters(), hyper.grad_clip)

        self.encoder_optim.step()
        self.decoder_optim.step()

        if epoch%1000 == 0:
            self.encoder_optim = self.lr_decay(self.encoder_optim)
            self.decoder_optim = self.lr_decay(self.decoder_optim)

        # todo save

        # todo load

        # todo conditional generation
        # This uses z from a trained model encoder, but feeds a sample image into the decoder
        # Using the input, the model "reconstructs" the image
        # Thus, not deterministic, but random
        # Temperature parameter controls randomness

        # todo unconditional generation
        # hidden/cell are initialized to zero and no z vector is used
        # encoder is _not_ trained
        # Can sample images and vary temp to get more varied output


    def lr_decay(self, optimizer):
        """Decay learning rate by a factor of lr_decay"""
        for param_group in optimizer.param_groups:
            if param_group['lr'] > hyper.min_lr:
                param_group['lr'] *= hyper.lr_decay
        return optimizer

    def bivariate_normal_pdf(self, dx, dy):
        z_x = ((dx - self.mu_x) / self.sigma_x) ** 2
        z_y = ((dy - self.mu_y) / self.sigma_y) ** 2
        z_xy = (dx - self.mu_x) * (dy - self.mu_y) / (self.sigma_x * self.sigma_y)
        z = z_x + z_y - 2 * self.rho_xy * z_xy
        exp = torch.exp(-z / (2 * (1 - self.rho_xy ** 2)))
        norm = 2 * np.pi * self.sigma_x * self.sigma_y * torch.sqrt(1 - self.rho_xy ** 2)
        return exp / norm

    def reconstruction_loss(self, mask, dx, dy, p, epoch):
        pdf = self.bivariate_normal_pdf(dx, dy)
        LS = -torch.sum(mask * torch.log(1e-5 + torch.sum(self.pi * pdf, 2))) / float(sketches.Nmax * hyper.batch_size)
        LP = -torch.sum(p * torch.log(self.q)) / float(sketches.Nmax * hyper.batch_size)
        return LS + LP

    def kullback_leibler_loss(self):
        LKL = -0.5 * torch.sum(1 + self.sigma - self.mu ** 2 - torch.exp(self.sigma))/ float(hyper.Nz * hyper.batch_size)
        KL_min = Variable(torch.Tensor([hyper.KL_min]).cuda()).detach()
        return hyper.wKL * self.eta_step * torch.max(LKL, KL_min)


sketches = sketchLoader()