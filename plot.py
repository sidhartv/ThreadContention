from sys import argv
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm
def plot_figure(err, sample_nums, plot_filename):
	plt.figure()
	plt.plot(sample_nums, err)

	avg = np.mean(err)
	p99 = np.percentile(err, 1)
	plt.axhline(avg, color='k', linestyle='dashed')
	_, max_ = plt.xlim()
	plt.text(max_ - max_/10, avg + avg/10,
		'Mean: {:.2f}'.format(avg))

	plt.text(max_ - max_/10, p99 + p99/10,
		'p99: {:.2f}'.format(p99))

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
