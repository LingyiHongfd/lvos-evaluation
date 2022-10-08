#!/usr/bin/env python
from ast import arg
import os
import sys
from time import time
import argparse

import numpy as np
import pandas as pd
from lvos.evaluation_mp import LVOSEvaluation as LVOSEvaluation_MP
from lvos.evaluation import LVOSEvaluation as LVOSEvaluation_SP

default_llvos_path = r'/home/hongly/LLVOS/VTGT'

time_start = time()
parser = argparse.ArgumentParser()
parser.add_argument('--lvos_path', type=str, help='Path to the LVOS folder containing the JPEGImages, Annotations, '
                                                   'ImageSets, Annotations_unsupervised folders',
                    required=False, default=default_llvos_path)
parser.add_argument('--set', type=str, help='Subset to evaluate the results', default='val')
parser.add_argument('--mp_nums', type=int, default=1, help='Multiple process numbers',)

parser.add_argument('--task', type=str, help='Task to evaluate the results', default='semi-supervised',)
parser.add_argument('--results_path', type=str, help='Path to the folder containing the sequences folders',
                    required=True)
args, _ = parser.parse_known_args()
if args.mp_nums<=1:
    args.mp_nums=1
    LVOSEvaluation=LVOSEvaluation_SP
    print(f'Evaluating with single processing.')
else:
    LVOSEvaluation=LVOSEvaluation_MP
    print(f'Evaluating with multiple processing with {args.mp_nums} processes.')




csv_name_global = f'global_results-{args.set}.csv'
csv_name_per_sequence = f'per-sequence_results-{args.set}.csv'

# Check if the method has been evaluated before, if so read the results, otherwise compute the results
csv_name_global_path = os.path.join(args.results_path, csv_name_global)
csv_name_per_sequence_path = os.path.join(args.results_path, csv_name_per_sequence)
if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
    print('Using precomputed results...')
    table_g = pd.read_csv(csv_name_global_path)
    table_seq = pd.read_csv(csv_name_per_sequence_path)
else:
    print(f'Evaluating sequences for the {args.task} task...')
    # Create dataset and evaluate
    if args.mp_nums<=1:
        dataset_eval = LVOSEvaluation(llvos_root=args.lvos_path, task=args.task, gt_set=args.set)
    else:
        dataset_eval = LVOSEvaluation(llvos_root=args.lvos_path, task=args.task, gt_set=args.set, mp_procs=args.mp_nums)

    metrics_res = dataset_eval.evaluate(args.results_path)
    J, F ,V = metrics_res['J'], metrics_res['F'], metrics_res['V']

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay', 'V_Mean']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"]), np.mean(V["M"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'w') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")
    print(f'Global results saved in {csv_name_global_path}')

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean', 'V-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    V_per_object = [V['M_per_object'][x] for x in seq_names]

    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object, V_per_object)), columns=seq_measures)
    with open(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")
    print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

# Print the results
sys.stdout.write(f"--------------------------- Global results for {args.set} ---------------------------\n")
print(table_g.to_string(index=False))
sys.stdout.write(f"\n---------- Per sequence results for {args.set} ----------\n")
print(table_seq.to_string(index=False))
total_time = time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))
sys.stdout.write('\n')
