import os, sys, time, re
import numpy as np
import pandas as pd
from hmmlearn import hmm
from pm4py.objects.petri.importer import pnml as pnml_importer
from pm4py.objects.petri import reachability_graph as reachability
from pm4py.visualization.transition_system.util import visualize_graphviz  as vis_graphviz


DATASET_NAME = 'p1-a8-g4'
DATASET_DIR = '../data/{}'.format(DATASET_NAME)


reachability.staterep = lambda name: re.sub(r'\W+', '_', name[2:-2])


if __name__ == '__main__':
    net_fname = '{}.pnml'.format(DATASET_NAME)
    net_fp = os.path.join(DATASET_DIR, net_fname)
    net, init_marking, final_marking = pnml_importer.import_net(net_fp)

    for place in net.places:
        place.name = 'p{}'.format(place.name)

    start_time = time.time()
    reach_graph = reachability.construct_reachability_graph(net, init_marking)
    end_time = time.time()
    print('Computing reachability graph took: {:.2f}s'.format(end_time - start_time))

    print('Number of distinct states: {}'.format(len(reach_graph.states)))

    ts_viz = vis_graphviz.visualize(reach_graph)
    ts_viz.save('{}-rc-graph.dot'.format(DATASET_NAME), '.')
