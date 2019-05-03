import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import argparse
import numpy as np

model_params = \
    {"input_size": 1,
     "output_size": 1,
     "hidden_size": 1,
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

        return linear_out

def parse_line(line):
    tokens = first_line.strip().split(" ")
    thread_time = int(tokens[0])
    lock_id = int(tokens[2])
    event_type = int(tokens[3])

    return thread_time, lock_id, event_type


def merge_same_time_data(fp):
    # target = np.zeros((num_locks, num_actions))
    
    first_line_tokens = first_line.strip().split(" ")

    thread_time = int(first_line_tokens[0])
    lock_id = int(first_line_tokens[2])-1
    event_type = int(first_line_tokens[3])-1

    # loop until next line has different timestamp
    while True:
        next_line = fp.readline()
        tokens = next_line.strip().split(" ")
        # handle case that we are at end of trace (num locks line)
        if len(tokens) == 1:
            return thread_time, target, None

        cur_time = int(tokens[0])
        # if next line has same timestamp, update target matrix and continue
        if cur_time == thread_time:
            # update target matrix
            lock_id = int(tokens[2])-1
            event_type = int(tokens[3])-1
        else:
            break

    # since we've already read the next line, we need to return it
    # so that next call to function can process it
    return thread_time, next_line




def train(lstm, log_file, lock_id=1, event_type=1, lr=0.01):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr)
    cur_time = 0
    old delta = 0
    new_delta = 0

    with open(log_file, "r") as fp:
        for line in fp:
            thread_time, line_lock_id, line_event_type = parse_line(line)
            if lock_id == line_lock_id and event_type == line_event_type:
                # we care about this event    
                if old_delta == -1:
                    # first time
                    old_delta = thread_time
                else:
                    # use delta between last two events to predict next delta
                    pred = model(torch.Tensor([old_delta]))
                    # calculate real delta between current and prev events
                    new_delta = thread_time - cur_time
                    
                    loss = loss_fn(pred, new_delta)
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    old_delta = new_delta

                cur_time = thread_time


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
                        default=0.1, help="The learning rate")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    log_file = args.log_file
    lock_id = args.lock_id
    lr = args.lr
    num_locks = get_model_dimensions(log_file)
    if lock_id <= 0 or lock_id > num_locks:
        print "Invalid lock ID"
        return

    input_size = model_params["input_size"]
    output_size = model_params["output_size"]
    hidden_size = model_params["hidden_size"]
    batch_size = model_params["batch_size"]
    num_layers = model_params["num_layers"]

    model = LSTM(input_size, hidden_size, batch_size=batch_size, 
            output_dim=output_size, num_layers=num_layers)

    # randomly initialize hidden weights
    model.init_hidden()

    trained_model = train(model, log_file, lock_id, event_type, lr)
