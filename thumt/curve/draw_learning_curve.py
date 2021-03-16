import matplotlib.pyplot as plt
import argparse
import os
plt.switch_backend('agg')
parser = argparse.ArgumentParser('draw learning curve')

parser.add_argument('--eval_history', type=str)
parser.add_argument('--record', type=str)
parser.add_argument('--figname', type=str)

args = parser.parse_args()

if args.figname == None:
	raise Exception('--figname should not be None')

def draw_bleu_history(path, figname):
	if not os.path.exists(path):
		raise Exception('Path %s does not exist' % path)

	# extract #iter and bleu score
	iters = []
	bleus = []
	with open(path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			if line == '\n':
				break
			line_split = line.split(' ')
			bleu = float(line_split[1]) * 100
			iter_no = int(line_split[0].split('-')[1][:-2])
			iters.append(iter_no)
			bleus.append(bleu)

	# draw learning curve
	plt.plot(iters, bleus)
	plt.ylabel('BLEU')
	plt.xlabel('Iteration No.')
	plt.savefig(figname + '.png')

if __name__ == '__main__':
	if args.eval_history != None:
		draw_bleu_history(args.eval_history, args.figname)

