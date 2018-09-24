import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from utils.utils_rnn import Dictionary, Corpus

device = 'cpu'

# Hyper-parameters
embed_size = 128
hidden_size = 1024
num_layers = 2
num_epochs = 40
batch_size = 30
seq_length = 20
learning_rate = 0.0025

corpus = Corpus()
word_ids = corpus.get_data("data/train.txt", batch_size)
vocab_size = len(corpus.dictionary)
number_batches = word_ids.size(1) // seq_length

class RNN(nn.Module):
    '''
    Multi layer LSTM, follows structure:
    Encoder -> LSTM -> Decoder
    '''
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout = 0.5):
        super(RNN, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.drop(self.embeddings(x))
        out, (h, c) = self.lstm(x, h)
        out = self.drop(out)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.linear(out)
        return out, (h, c)

model = RNN(vocab_size, embed_size, hidden_size, num_layers).to(device)

criterion = nn.CrossEntropyLoss() # Since we have to report perplexity
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    Input: States
    Output Detached States
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

# model.load_state_dict(torch.load('model.ckpt'))
# model.lstm.flatten_parameters()

def get_batch(data, i, length):
    '''
    Slicing the batches.
    Doesn't slow down process. Maybe I'll look into torchtext wrappers.
    '''
    inputs = data[:, i: i + length].to(device)
    targets = data[:, (i + 1): (i + 1) + length].to(device)
    return inputs, targets

def validate(model):
    '''
    Validation epoch, to run after every training epoch.
    '''
    model.eval()
    eval_batch_size = 1
    states = (torch.zeros(num_layers, eval_batch_size, hidden_size).to(device),
              torch.zeros(num_layers, eval_batch_size, hidden_size).to(device))
    total_loss = 0
    count = 0
    test_ids = corpus.get_data("data/valid.txt", eval_batch_size)
    num_batches = test_ids.size(1) // seq_length

    for i in range(0, test_ids.size(1) - seq_length, seq_length):

        inputs, targets = get_batch(test_ids, i, seq_length)
        states = repackage_hidden(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))
        total_loss += len(inputs) * loss.item()
        count += 1
        current_loss = total_loss / count

        step = (i + 1) // seq_length
        if step % 100 == 0:
            print ('Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                   .format(step, num_batches, current_loss, np.exp(current_loss)))


###### Training Loop ####################################################

try:
    for epoch in range(num_epochs):
        model.train()
        # LSTM has 2 states!
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))

        for i in range(0, word_ids.size(1) - seq_length, seq_length):
            inputs, targets = get_batch(word_ids, i, seq_length)

            states = repackage_hidden(states)
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape(-1))

            model.zero_grad()
            loss.backward()

            # To avoid gradient explosion
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            step = (i + 1) // seq_length

            # Logging errors after 100th step
            if step % 100 == 0:
                print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                       .format(epoch+1, num_epochs, step, number_batches, loss.item(), np.exp(loss.item())))
        validate(model)

except KeyboardInterrupt:
    print('Exiting from training early')

torch.save(model.state_dict(), 'model.ckpt')

###### Testing Loop ####################################################

model.load_state_dict(torch.load('model.ckpt'))
model.lstm.flatten_parameters()
model.eval()
eval_batch_size = 1
states = (torch.zeros(num_layers, eval_batch_size, hidden_size).to(device),
          torch.zeros(num_layers, eval_batch_size, hidden_size).to(device))
total_loss = 0
count = 0
test_ids = corpus.get_data("data/test.txt", eval_batch_size)
num_batches = test_ids.size(1) // seq_length

for i in range(0, test_ids.size(1) - seq_length, seq_length):
    inputs, targets = get_batch(test_ids, i, seq_length)

    states = repackage_hidden(states)
    outputs, states = model(inputs, states)
    loss = criterion(outputs, targets.reshape(-1))
    total_loss += len(inputs) * loss.item()
    count += 1
    current_loss = total_loss / count

    step = (i + 1) // seq_length
    if step % 100 == 0:
        print ('Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
               .format(step, num_batches, current_loss, np.exp(current_loss)))
