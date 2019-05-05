import keras
import numpy as np
import argparse


def construct_model(learning_rate):
	input0 = keras.layers.Input(shape=(1,1))

	lstm0 = keras.layers.LSTM(32, input_shape=(1,1), return_sequences=True)(input0)
	lstm1 = keras.layers.LSTM(32)(lstm0)
	dense1 = keras.layers.Dense(32)(lstm1)

	fc_delta = keras.layers.Dense(1)(dense1)

	model = keras.models.Model(inputs=[input0], outputs=[fc_delta])

	opt = keras.optimizers.Adam(lr=learning_rate)
	model.compile(loss=['mean_squared_error'], optimizer=opt)
	return model

def parse_line(line):
    tokens = line.strip().split(" ")

    if len(tokens) == 1:
        return -1, -1, -1

    thread_time = int(tokens[0])
    lock_id = int(tokens[2])
    event_type = int(tokens[3])

    return thread_time, lock_id, event_type

def parse_trace(trace_file, filter_lock_id, filter_event):
	trace = []
	last_thread_time = 0
	with open(trace_file, 'r') as fp:
		for line in fp:
			thread_time, lock_id, event_type = parse_line(line)
			if lock_id != filter_lock_id or event_type != filter_event:
				continue
			delta = thread_time - last_thread_time
			last_thread_time = thread_time

			rec = np.array(delta).reshape(1,1)

			trace.append(rec)

	x = np.array(trace[:-1])
	y = np.array(trace[1:])

	return x, y


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
    parser.add_argument('--lock-id', dest='lock_id', type=int,
                        default=1, help="The lock ID to analyze")
    parser.add_argument('--event-type', dest='event_type', type=int,
                        default=1, help="The event type to analyze and predict timing of")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.01, help="The learning rate")
    parser.add_argument('--epochs', dest='epochs', type=float,
                        default=150, help="Number of epochs")

    return parser.parse_args()

def main():
	args = parse_args()
	log_file = args.log_file
	lock_id = args.lock_id
	event_type = args.event_type
	lr = args.lr
	epochs = args.epochs
	model = construct_model(lr)
	print("[INFO] Model constructed")
	x,y = parse_trace(log_file, lock_id, event_type)

	x = x.reshape((x.shape[0], 1, x.shape[1]))
	y = y.reshape((y.shape[0], y.shape[1]))

	print("[INFO] Trace parsed")
	model.fit(x, y, epochs=epochs)

	



main()