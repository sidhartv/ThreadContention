import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import argparse
import numpy as np

model_params = \
    {"input_size": 1,
     "num_actions": 2,
     "hidden_size": 1,
     "num_layers": 1,
     "batch_size": 1
    }

# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, num_locks, num_actions=15,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.num_locks = num_locks

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        lin_output_dim = num_actions * num_locks
        self.linear = nn.Linear(self.hidden_dim, lin_output_dim)
        self.sigmoid = nn.Sigmoid()

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
        y_pred = self.sigmoid(linear_out[-1].view(self.batch_size, -1))

        # TODO: verify correct dimensions
        return y_pred.view(num_locks, num_actions)


def merge_same_time_data(fp, first_line, num_locks, num_actions):
    target = np.zeros((num_locks, num_actions))
    
    first_line_tokens = first_line.strip().split(" ")
    thread_time = int(first_line_tokens[0])
    object_id = int(first_line_tokens[2])-1
    event_type = int(first_line_tokens[3])-1
    target[object_id][event_type] = 1

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
            object_id = int(tokens[2])-1
            event_type = int(tokens[3])-1
            target[object_id][event_type] = 1
        else:
            break

    # since we've already read the next line, we need to return it
    # so that next call to function can process it
    return thread_time, torch.tensor(target, requires_grad=True).float(), next_line


def train(lstm, log_file, lr=0.01):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr)
    cur_time = 1

    num_locks = lstm.num_locks
    num_actions = lstm.num_actions

    nop_target = torch.zeros(num_locks, num_actions)
    with open(log_file, "r") as fp:
        next_line = fp.readline()
        # loop until end of file
        while next_line:
            data_time, data_target, next_line = merge_same_time_data(fp, next_line,
                num_locks, num_actions)

            while cur_time < data_time:
                pred = model(torch.Tensor([cur_time]))
                loss = loss_fn(pred, nop_target)
                print cur_time, loss.item()
                loss.backward(retain_graph=True)
                optimizer.step()
                cur_time += 1
                # raw_input()

            pred = model(torch.Tensor([data_time]))
            loss = loss_fn(pred, data_target)
            print data_time, data_target, loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            cur_time += 1
            # raw_input()

            if cur_time >= 32000:
                return lstm

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
    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.1, help="The learning rate.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    log_file = args.log_file
    lr = args.lr
    num_locks = get_model_dimensions(log_file)

    input_size = model_params["input_size"]
    num_actions = model_params["num_actions"]
    hidden_size = model_params["hidden_size"]
    batch_size = model_params["batch_size"]
    num_layers = model_params["num_layers"]

    model = LSTM(input_size, hidden_size, batch_size=batch_size, 
        num_locks=num_locks, num_actions=num_actions, num_layers=num_layers)

    model.init_hidden()

    # lstm = nn.LSTM(input_size=model_params["input_dim"], 
    #         hidden_size=output_size,
    #         num_layers=model_params["num_layers"],
    #         bias=True,
    #         batch_first=False,
    #         dropout=0,
    #         bidirectional=False
    #     )

    trained_model = train(model, log_file, lr)

