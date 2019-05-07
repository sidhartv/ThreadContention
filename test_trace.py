import keras
import argparse
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-file', dest='trace_file', type=str, default=None)
    parser.add_argument('--lock-id', dest='lock_id', type=str, default='0x00')
    parser.add_argument('--event-id', dest='event_id', type=int, default=-1)
    parser.add_argument('--model-file', dest='model_file', type=str,
                        default=None, help="Filepath of existing model")

    return parser.parse_args()


def parse_line(line):
    tokens = line.strip().split(" ")

    if len(tokens) == 1:
        return -1, -1, -1

    thread_time = int(tokens[0])
    lock_id = int(tokens[2], 16)
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
	if len(trace) < threshold:
		return [], []
	x = np.array(trace[:-1])
	y = np.array(trace[1:])

	return x, y


def main():
	args = parse_args()
	trace_file = args.trace_file
	lock_id = args.lock_id
	event_id = args.event_id
	model_file = args.model_file

	x, y = parse_trace(trace_file, lock_id, event_id)
	model = keras.models.load_model(model_file)

	model.evaluate(x,y)

main()
