import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_cnn import read_words, create_batches, to_var
import numpy as np

class GatedConvNet(nn.Module):

    def __init__(self, seq_len, vocab_size, embed_size, n_layers, kernel_size, num_filters, res_block_count):
        super(GatedConvNet, self).__init__()

        # create embedding matrix
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # first entry of convolutions
        self.first_conv = nn.Conv2d(1, num_filters, kernel_size, padding=(2, 0))

        # add bias to first convolutional layer
        self.bias_first_conv = nn.Parameter(torch.randn(1, num_filters, 1, 1))

        # first entry of convolutions with gate
        self.first_conv_gated = nn.Conv2d(1, num_filters, kernel_size, padding=(2, 0))

        # add bias to first convolutional layer with gate
        self.bias_first_conv_gated = nn.Parameter(torch.randn(1, num_filters, 1, 1))

        # define function for convolution stack
        self.convolve = nn.ModuleList([nn.Conv2d(num_filters, num_filters, (kernel_size[0], 1), padding=(2, 0)) for _ in range(n_layers)])

        # define bias for convolution stack
        self.bias_convolve = nn.ParameterList([nn.Parameter(torch.randn(1, num_filters, 1, 1)) for _ in range(n_layers)])

        # define function for convolution stack with gate
        self.convolve_gate = nn.ModuleList([nn.Conv2d(num_filters, num_filters, (kernel_size[0], 1), padding=(2, 0)) for _ in range(n_layers)])

        # define bias for convolution stack with gate
        self.bias_convolve_gate = nn.ParameterList([nn.Parameter(torch.randn(1, num_filters, 1, 1)) for _ in range(n_layers)])

        # final decoder to vocab size
        self.fc = nn.Linear(num_filters * seq_len, vocab_size)
        self.res_block_count = res_block_count

    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)
        x = self.embedding(x)

        # add another dimension
        x = x.unsqueeze(1)

        # do the convolution and add the bias
        without_gate = self.first_conv(x)
        without_gate += self.bias_first_conv.repeat(1, 1, seq_len, 1)

        # repeating the bias tensor seq_len times

        with_gate = self.first_conv_gated(x)
        with_gate += self.bias_first_conv_gated.repeat(1, 1, seq_len, 1)

        h = without_gate * F.sigmoid(with_gate)
        res_input = h

        # applying idea of resnets here

        for i, (conv, conv_gate) in enumerate(zip(self.convolve, self.convolve_gate)):
            # convolutions for states without gate
            A = conv(h) + self.bias_convolve[i].repeat(1, 1, seq_len, 1)

            # convolutions for states with gate
            B = conv_gate(h) + self.bias_convolve_gate[i].repeat(1, 1, seq_len, 1)

            # gating
            h = A * F.sigmoid(B)

            # adding residual connections
            if i % self.res_block_count == 0: # size of each residual block
                h += res_input
                res_input = h

        h = h.view(batch_size, -1)
        out = self.fc(h)
        out = F.log_softmax(out, dim = 1)

        return out

# Hyperparameters

vocab_size      = 2000
embed_size      = 200
seq_len         = 17
n_layers        = 10
kernel_size     = (5, embed_size)
num_filters     = 64
res_block_count = 5
batch_size      = 64

words = read_words('./data', seq_len, kernel_size[0])

# encoding matrix, extract most popular words
word_counter = collections.Counter(words).most_common(vocab_size - 1)
vocab = [w for w, _ in word_counter]

# assign word IDs
word_ids = dict((w, i) for i, w in enumerate(vocab, 1))
word_ids['<unk>'] = 0

# read files
data = [word_ids[w] if w in word_ids else 0 for w in words]
data = create_batches(data, batch_size, seq_len)
split_idx = int(len(data) * 0.8)
training_data = data[:split_idx]
test_data = data[split_idx:]


def train(model, data, test_data, optimizer, loss_fn, epochs = 10):
    '''
    Training Loop
    '''
    model.train()
    for epoch in range(epochs):
        print('Epoch', epoch)
        random.shuffle(data)
        for batch_ct, (X, Y) in enumerate(data):
            X = to_var(torch.LongTensor(X))
            Y = to_var(torch.LongTensor(Y))
            pred = model(X)
            loss = loss_fn(pred, Y)

            if batch_ct % 100 == 0:
                print('Training Loss: {:.4f} Perplexity: {:.4f}'.format(loss.item(), np.exp(loss.item())))

            model.zero_grad()
            loss.backward()
            optimizer.step()
        print('Test set performance', epoch)
        test(model, test_data)

def test(model, data):
    '''
    Testing Loop
    '''
    model.eval()
    counter = 0
    correct = 0
    losses = 0.0
    for batch_ct, (X, Y) in enumerate(data):
        X = to_var(torch.LongTensor(X))
        Y = to_var(torch.LongTensor(Y))
        pred = model(X) #
        loss = loss_fn(pred, Y)
        losses += torch.sum(loss).item() # Accumulative averages
        _, pred_ids = torch.max(pred, 1)
        print('Loss: {:.4f}'.format(loss.item()))
        correct += torch.sum(pred_ids == Y).item()
        counter += 1

    loss = losses/counter
    ppl = np.exp(loss)
    print('Test Loss: {:.4f} Perplexity: {:.4f}'.format(losses/counter, ppl))

model = GatedConvNet(seq_len, vocab_size, embed_size, n_layers, kernel_size, num_filters, res_block_count)
optimizer = torch.optim.Adadelta(model.parameters())
loss_fn = nn.CrossEntropyLoss()
train(model, training_data, test_data, optimizer, loss_fn)
