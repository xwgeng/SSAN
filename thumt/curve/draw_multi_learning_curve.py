import matplotlib.pyplot as plt
import argparse
import os
plt.switch_backend('agg')

parser = argparse.ArgumentParser('draw multiple learning curves')
parser.add_argument('--eval_histories', nargs='+', type=str, required=True)
parser.add_argument('--figname', required=True)

args = parser.parse_args()

if args.figname == None:
    raise Exception('figname should be given!')

def draw_multi_bleu_histories(paths, figname):
    bleu_lst = []
    iter_no_lst = []
    # extract bleus and #iter
    for path in paths:
        with open(path, 'r') as f:
            bleus = []
            iter_nos = []
            lines = f.readlines()
            for line in lines:
                if line == '\n':
                    break
                line_split = line.split(' ')
                bleu = float(line_split[1]) * 100
                iter_no = int(line_split[0].split('-')[1][:-2])
                bleus.append(bleu)
                iter_nos.append(iter_no)

            bleu_lst.append(bleus)
            iter_no_lst.append(iter_nos)
    # draw figure
    color_lst = ['g', 'r', 'b', 'y']
    curves = []
    labels = ['small', 'middle', 'big', 'big_gpu2']
    for fig_idx in range(len(bleu_lst)):
        bleus = bleu_lst[fig_idx]
        iter_nos = iter_no_lst[fig_idx]
        curve, = plt.plot(iter_nos, bleus, color_lst[fig_idx], label=labels[fig_idx])
        curves.append(curve)
    # print(curves)
    plt.ylabel('BLEU')
    plt.xlabel('Iteration No.')
    plt.legend(handles=curves)
    plt.savefig(figname + '.png')


if __name__ == '__main__':
    draw_multi_bleu_histories(args.eval_histories, args.figname)
