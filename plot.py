from sys import argv
import numpy as np
def plot_figure(errs, sample_nums, plot_filename):
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
	file = argv[1]
	plot_filename = file[:-3] + '.png'

	npfile = np.load(file)
	errs = npfile['errs']
	sample_nums = npfile['sample_nums']
	plot_figure(errs, sample_nums, plot_filename)

main()
