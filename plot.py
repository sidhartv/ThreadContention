from sys import argv
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm
def plot_figure(errs, sample_nums, plot_filename, ylim):
	median = errs[-1]
	err = errs[:-1]
	plt.figure()
	if len(sample_nums) > len(err):
		sample_nums = sample_nums[:-1]

	plt.plot(sample_nums, err)

	avg = np.mean(err)
	p99 = np.percentile(err, 90)
	plt.axhline(median, color='k', linestyle='dashed', label='median')
	if ylim > 0:
		plt.ylim((0,ylim))
	else:
		plt.ylim(bottom=0)

	_, max_x = plt.xlim()
	_, max_y = plt.ylim()

	plt.text(10, max_y,
                'p90: {:.2f}\n median delta-t: {}'.format(p99, median))
	plt.xlabel('Progression of trace')
	plt.ylabel('Error in microseconds')
	plt.savefig(plot_filename)
	plt.close()

def main():
	directory = argv[1]
	ylim = int(argv[2])
	if directory[-1] != '/':
		directory += '/'
	files = glob(directory + '*.npz')
	for file in tqdm(files):
		plot_filename = file[:-3] + 'png'

		npfile = np.load(file)
		errs = npfile['err']
		sample_nums = npfile['sample_num']
		plot_figure(errs, sample_nums, plot_filename, ylim)

main()
