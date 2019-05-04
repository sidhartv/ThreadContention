import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import argparse
import numpy as np

from tqdm import tqdm

model_params = \
    {"input_size": 1,
     "output_size": 3,
     "hidden_size": 10,
     "num_layers": 1,
     "batch_size": 1
    }

# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, inputs):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(inputs.view(len(inputs), self.batch_size, -1), self.hidden)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        linear_out = self.linear(lstm_out[-1].view(self.batch_size, -1))

        return linear_out.view(self.output_dim)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def parse_line(line):
    tokens = line.strip().split(" ")

    if len(tokens) == 1:
        return -1, -1, -1

    thread_time = int(tokens[0])
    lock_id = int(tokens[2])
    event_type = int(tokens[3])

    return thread_time, lock_id, event_type


# def merge_same_time_data(fp):
#     # target = np.zeros((num_locks, num_actions))
    
#     first_line_tokens = first_line.strip().split(" ")

#     thread_time = int(first_line_tokens[0])
#     lock_id = int(first_line_tokens[2])-1
#     event_type = int(first_line_tokens[3])-1

#     # loop until next line has different timestamp
#     while True:
#         next_line = fp.readline()
#         tokens = next_line.strip().split(" ")
#         # handle case that we are at end of trace (num locks line)
#         if len(tokens) == 1:
#             return thread_time, target, None

#         cur_time = int(tokens[0])
#         # if next line has same timestamp, update target matrix and continue
#         if cur_time == thread_time:
#             # update target matrix
#             lock_id = int(tokens[2])-1
#             event_type = int(tokens[3])-1
#         else:
#             break

#     # since we've already read the next line, we need to return it
#     # so that next call to function can process it
#     return thread_time, next_line




def train(lstm, log_file, lr=0.01, epochs=150):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr)

    lines = []
    with open(log_file, "r") as fp:
        for line in fp:
            lines.append(line)

    for epoch in range(epochs):
        epoch_loss = 0
        cur_time = 0

        old_lock_id = -1
        old_event_type = -1
        old_delta = -1

        new_lock_id = 0
        new_event_type = 0
        new_delta = 0

        for line in tqdm(lines):
            thread_time, line_lock_id, line_event_type = parse_line(line)
            # if lock_id == line_lock_id and event_type == line_event_type:
            # we care about this event    
            if old_delta == -1:
                # first time
                old_delta = thread_time
                old_lock_id = line_lock_id
                old_event_type = line_event_type

            else:
                # t = torch.Tensor([old_delta, old_lock_id, old_event_type])
                # print t.view(3, 1, 1).size()

                # use delta between last two events to predict next delta
                pred = lstm(torch.Tensor([old_delta, old_lock_id, old_event_type]))
                lstm.hidden = repackage_hidden(lstm.hidden)

                # calculate real delta between current and prev events
                new_lock_id = line_lock_id
                new_event_type = line_event_type
                new_delta = thread_time - cur_time
                
                loss = loss_fn(pred, torch.Tensor([new_delta, new_lock_id, new_event_type]))
                #print cur_time, thread_time, old_delta, new_delta, pred, loss.item()

                epoch_loss += loss.item()

                # loss.backward(retain_graph=True)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                old_lock_id = new_lock_id
                old_event_type = new_event_type
                old_delta = new_delta

                cur_time = thread_time

        print('Epoch #' + str(epoch) + ' complete. Loss = ' + str(epoch_loss / len(lines)))


    return lstm


def get_model_dimensions(log_file):
    with open(log_file, "r") as fp:
        fp.seek(-2, os.SEEK_END)
        while fp.read(1) != b'\n':
            fp.seek(-2, os.SEEK_CUR)

        num_locks = int(fp.readline().decode())

    return num_locks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', dest='log_file', type=str,
                        default=None, help="Filepath of log to build model of")
    parser.add_argument('--lock_id', dest='lock_id', type=int,
                        default=1, help="The lock ID to analyze")
    parser.add_argument('--event_type', dest='event_type', type=int,
                        default=1, help="The event type to analyze and predict timing of")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.01, help="The learning rate")
    parser.add_argument('--epochs', dest='epochs', type=float,
                        default=150, help="Number of epochs")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    log_file = args.log_file
    lock_id = args.lock_id
    event_type = args.event_type
    lr = args.lr
    num_locks = get_model_dimensions(log_file)
    if lock_id <= 0 or lock_id > num_locks:
        print "Invalid lock ID"
        exit()

    input_size = model_params["input_size"]
    output_size = model_params["output_size"]
    hidden_size = model_params["hidden_size"]
    batch_size = model_params["batch_size"]
    num_layers = model_params["num_layers"]

    model = LSTM(input_size, hidden_size, batch_size=batch_size, 
            output_dim=output_size, num_layers=num_layers)

    # randomly initialize hidden weights
    model.init_hidden()

    trained_model = train(model, log_file, lr)

