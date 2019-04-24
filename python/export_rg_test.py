import numpy as np
import pandas as pd
import time, os, argparse, subprocess
import multiprocessing as mp

import hmmconf
from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.visualization.transition_system import util
from pm4py.visualization.transition_system.util import visualize_graphviz  as vis_graphviz


MODEL_DIR = os.path.join('..', 'data', 'BPM2018', 'correlation-tests', 'models')
LOG_DIR = os.path.join('..', 'data', 'BPM2018', 'correlation-tests', 'logs', 'processed')
IMG_DIR = os.path.join('.', 'image', 'correlation-tests')
DOT_DIR = os.path.join(IMG_DIR, 'dot')
PNG_DIR = os.path.join(IMG_DIR, 'png')


if not os.path.isdir(IMG_DIR):
    os.mkdir(IMG_DIR)

if not os.path.isdir(DOT_DIR):
    os.mkdir(DOT_DIR)

if not os.path.isdir(PNG_DIR):
    os.mkdir(PNG_DIR)


logger = hmmconf.utils.make_logger(__file__)


def process_net(net, init_marking, final_marking):
    is_inv = lambda t: t.label is None
    inv_trans = list(filter(is_inv, net.transitions))
    print('Number of invisible transitions: {}'.format(len(inv_trans)))
    rg, inv_states = hmmconf.build_reachability_graph(net, init_marking, is_inv)
    is_inv = lambda t: t.name is None
    hmmconf.connect_inv_markings(rg, inv_states, is_inv)
    return rg


if __name__ == '__main__':
    start_all = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', action='store',
                        dest='netfiles',
                        help='List of net files to run')

    args = parser.parse_args()

    if args.netfiles is None:
        logger.info('Run as python ./correlation_test.py -f [netfiles]')
        exit(0)

    with open(args.netfiles, 'r') as f:
        netfiles = f.readlines()
    netfiles = list(map(lambda name: name.strip(), netfiles))

    logger.info('Running correlation test on: \n{}'.format(netfiles))

    for net_fname in netfiles:
        net_fp = os.path.join(MODEL_DIR, net_fname)
        net, init_marking, final_marking = pnml_importer.import_net(net_fp)
        rg = process_net(net, init_marking, final_marking)
        ts_viz = vis_graphviz.visualize(rg)
        dot_fname = net_fname.replace('.pnml', '.dot')
        dot_fp = os.path.join(DOT_DIR, dot_fname)
        ts_viz.save(dot_fname, DOT_DIR)

        png_fname = net_fname.replace('.pnml', '.png')
        png_fp = os.path.join(PNG_DIR, png_fname)
        subprocess.run(['dot', '-Tpng', dot_fp, '-o', png_fp])
