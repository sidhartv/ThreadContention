import numpy as np
from sys import argv

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

	x = np.array(trace[:-1])

	return np.median(x)


file = argv[1]
lock = int(argv[2], 16)
event = int(argv[3])

print(parse_trace(file, lock, event))