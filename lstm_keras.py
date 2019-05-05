import keras
import numpy as np


def construct_model():
	input0 = keras.layers.Input(shape=(1,3))

	lstm0 = keras.layers.LSTM(32, input_shape=(1,3), return_sequences=True)(input0)
	lstm1 = keras.layers.LSTM(32)(lstm0)

	fc_delta = keras.layers.Dense(1)(lstm1)

	fc_event = keras.layers.Dense(2)(lstm1)
	soft_event = keras.layers.Activation('softmax')(fc_event)

	model = keras.models.Model(inputs=[input0], outputs=[fc_delta, soft_event])

	opt = keras.optimizers.Adam(lr=0.05)
	model.compile(loss=['mean_squared_error', 'categorical_crossentropy'], optimizer=opt)
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

			rec = np.zeros(3)
			rec[0] = delta
			rec[event_type] = 1

			trace.append(rec)

	x = np.array(trace[:-1])
	y = np.array(trace[1:])

	return x, y


model = construct_model()
print("[INFO] Model constructed")
x,y = parse_trace('../sysbench-trace/processed/thread_2513_p.log', 3, 2)

x = x.reshape((x.shape[0], 1, x.shape[1]))
y = y.reshape((y.shape[0], y.shape[1]))

print("[INFO] Trace parsed")
model.fit(x, [y[:,0], y[:,1:3]], epochs=2000)