from sys import argv
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm
def plot_figure(errs, sample_nums, plot_filename):
	median = errs[-1]
	errs = errs[:-1]
	plt.figure()
	plt.plot(sample_nums, err)

	avg = np.mean(err)
	p99 = np.percentile(err, 90)
	plt.axhline(p99, color='k', linestyle='dashed')
	plt.ylim((0,20))

	_, max_x = plt.xlim()
	_, max_y = plt.ylim()

	plt.text(max_x - max_x/7, max_y + max_y/15,
                'p90: {:.2f}\n median delta-t: {}'.format(p99, median))
	plt.xlabel('Progression of trace')
	plt.ylabel('Error in microseconds')
	plt.savefig(plot_filename)
	plt.close()

def main():
	directory = argv[1]
	if directory[-1] != '/':
		directory += '/'
	files = glob(directory + '*.npz')
	for file in tqdm(files):
		plot_filename = file[:-3] + 'png'

		npfile = np.load(file)
		errs = npfile['err']
		sample_nums = npfile['sample_num']
		plot_figure(errs, sample_nums, plot_filename)

main()
