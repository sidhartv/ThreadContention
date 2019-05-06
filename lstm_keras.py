import keras
import numpy as np
import argparse

import multiprocessing 


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
    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.01, help="The learning rate")
    parser.add_argument('--epochs', dest='epochs', type=float,
                        default=150, help="Number of epochs")

    return parser.parse_args()

def get_combos(trace_file):
	combos = set()
	with open(trace_file, 'r') as fp:
		for line in fp:
			thread_time, lock_id, event_type = parse_line(line)
			combos.add((lock_id, event_type))
	return list(combos)


def train(x, y, model, epochs, lock_id, event_id, file_prefix):
	print('[INFO] Started process for (lock ' + str(lock_id) + ', event ' + str(event_id) + ')')
	x = x.reshape((x.shape[0], 1, x.shape[1]))
	y = y.reshape((y.shape[0], y.shape[1]))

	hist = model.fit(x, y, epochs=epochs, verbose=False)
	filename = file_prefix + '_' + str(lock_id) + '_' + str(event_id) + '.h5'
	model.save(filename)
	print('[INFO] Saved model to ' + filename)

def main():
	args = parse_args()
	trace_file = args.log_file
	lr = args.lr
	epochs = args.epochs

	combos = get_combos(trace_file)
	processes = []
	for combo in combos:
		(lock_id, event_id) = combo
		combo_x, combo_y = parse_trace(trace_file, lock_id, event_id)
		print('[INFO] Parsed trace for (lock ' + str(lock_id) + ', event ' + str(event_id) + ')')
		if len(combo_x) == 0:
			print('[INFO] Trace for (lock ' + str(lock_id) + ', event ' + str(event_id) + ') does not have enough samples')
			continue

		p = multiprocessing.Process(target=train, args=(combo_x, combo_y, construct_model(lr), epochs, lock_id, event_id, trace_file))
		processes.append(p)
		p.start()

	for p in processes:
		p.join()





main()