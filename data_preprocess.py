
# Format:

# TID
# Thread time (in nanoseconds)
# Object type (1=mutex, 2=condvar, 3=rwlock)
# Object address
# Event type

# Events are as follows:

# define EVENT_TYPE_MUTEX_LOCK (1)
# define EVENT_TYPE_MUTEX_UNLOCK (2)
# define EVENT_TYPE_MUTEX_TIMEDLOCK (3)
# define EVENT_TYPE_MUTEX_TRYLOCK (4)

# define EVENT_TYPE_COND_SIGNAL (5)
# define EVENT_TYPE_COND_BROADCAST (6)
# define EVENT_TYPE_COND_WAIT (7)
# define EVENT_TYPE_COND_TIMEDWAIT (8)

# define EVENT_TYPE_RWLOCK_RDLOCK (9)
# define EVENT_TYPE_RWLOCK_TRYRDLOCK (10)
# define EVENT_TYPE_RWLOCK_TIMEDRDLOCK (11)
# define EVENT_TYPE_RWLOCK_WRLOCK (12)
# define EVENT_TYPE_RWLOCK_TRYWRLOCK (13)
# define EVENT_TYPE_RWLOCK_TIMEDWRLOCK (14)
# define EVENT_TYPE_RWLOCK_UNLOCK (15)

# Preprocessing steps
#     - define timeslice length?
#         - 1 ms? normal scheduling interval?
#         - group data by timeslice into tensors
#     - label = next lock + action taken by thread 
#         - may be able to introduce nops into data
#             - new action that is doing nothing?
#         - or include time difference as input

# Model learning steps
#     - separate model per thread? 
#         - need to know number of threads
#         - too many models?
#         - how would models interact? how would we know if one thread will contend?
#             - we would get predictions based on temporal patterns and not thread interaction patterns
#         - but in data, time is based on thread time, not global time
#     - for each thread, output probability of each action for each lock in thread
#         - need to know number of locks beforehand
#         - most of the time, prediction should be to do nothing for each lock
#         - 
#     - Training:
#         - loss = MSE between predictions and actual action taken for each lock
#         - input = time
#         - label = lock ID, action (incorporates lock type)

#     - one model
#         - feed in all data points
#         - given thread ID, predict action for each lock it has
#         - OR predict actino for each lock for each thread
#             - num threads x num locks x num actions (huge space)
#             - means that loss will also be same space
#         - huge output space but we can get patterns of thread contention?

# Questions
#     - difference between trylock and lock?
#     - what is a timed lock
#     - focus on mutexes first? predictable program first?


# TODO:
#     - Preprocess
#         - lock addr -> lock ID
#         - bucket entries into time slices?
#             - still maintain original order
#         - insert nops for unfilled buckets
#     - Train
#         - label = lock ID, action
#         - loss = cross entropy loss
        


import os
import glob
import argparse 

no_lock_id = 0
next_object_id = 1
object_ids = dict()

interval_size_ns = 1000 # how to adjust and round the raw timestamps

def get_object_id(addr):
    global next_object_id
    if addr in object_ids:
        return object_ids[addr]
    else:
        object_ids[addr] = next_object_id
        next_object_id += 1
        return object_ids[addr]

def insert_nops(out_fp, prev_time, next_time):
    for t in range(prev_time + 1, next_time):
        out_fp.write("%d 0 0 0\n" % (t))

def parse_file(filepath, outpath):
    global object_ids

    filename = filepath.split('/')[-1]
    file_prefix = filename.split(".")[0] + "_" + filename.split(".")[1]
    file_type = filename.split(".")[2]
    out_filepath = os.path.join(outpath, file_prefix + "_p." + file_type)
    print out_filepath

    prev_time = 0

    with open(filepath, "r") as fp:
        with open(out_filepath, "w+") as out_fp:                                                                                                                                                                                                                                                                                                    
            for line in fp:        
                tokens = line.strip().split(" ")
                thread_time_ns = int(tokens[1])
                # round to buckets of size interval_size_ms
                thread_time_ns = thread_time_ns - (thread_time_ns % interval_size_ns)
                thread_time_us = thread_time_ns/interval_size_ns
                object_type = tokens[3]
		object_addr = tokens[4]
                object_id = get_object_id(tokens[4])
                event_type = tokens[5]

                # only comment in if you want to add nops 
                # insert_nops(out_fp, prev_time, thread_time_us)

                outline = "%d %s %s %s\n" % (thread_time_us, object_type, 
                    object_addr, event_type)
                # print(outline)
                out_fp.write(outline) 

                prev_time = thread_time_us

            # record number of locks in thread log
            #out_fp.write(str(len(object_ids)))


def parse_traces(path, outpath):
    global object_ids
    global next_object_id

    for filename in glob.glob(os.path.join(path, '*.log')):
    # for filename in os.listdir(path):
        object_ids = dict() # reset for next file
        next_object_id = 1
        print filename
        parse_file(filename, outpath)
        # exit()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', dest='input_dir', type=str,
        default=None, help="Directory of mutrace logs")
    parser.add_argument('--output-dir', dest='output_dir', type=str,
        default=None, help="Directory to place processed logs into")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    inpath = args.input_dir
    outpath = args.output_dir
    parse_traces(inpath, outpath)

        
