import pandas as pd
import numpy as np


from graphviz import Digraph
from . import pm_extra, utils


def petrinet2rg(net, init_marking, final_marking, staterep=pm_extra.default_staterep):
    is_inv = lambda t: t.label is None
    is_inv_rg = lambda t: t.name is None

    inv_trans = list(filter(is_inv, net.transitions))
    rg, inv_states = pm_extra.build_reachability_graph(net, init_marking, is_inv, staterep)
    pm_extra.connect_inv_markings(rg, inv_states, is_inv_rg)

    to_remove = list()
    for t in rg.transitions:
        if is_inv_rg(t):
            to_remove.append(t)

    for t in to_remove:
        rg.transitions.remove(t)
        t.from_state.outgoing.remove(t)
        t.to_state.incoming.remove(t)

    return rg


def petrinet2dot(net, init_marking, final_marking, highlight_marking_str=None, highlight_color='bisque', graph_attr=None):
    if highlight_marking_str:
        place_map = {str(p.name): p for p in net.places}
        highlight_marking = pm_extra.default_staterep_to_marking(highlight_marking_str, place_map)
    else:
        highlight_marking = None

    default_graph_attr = {
        'bgcolor': 'white',
        'rankdir': 'TB',
    }
    graph_attr = default_graph_attr if graph_attr is None else graph_attr
    g = Digraph(net.name, engine='dot', graph_attr=graph_attr, format='png')

    g.attr('node', shape='box')
    for t in net.transitions:
        if t.label is not None:
            g.node(str(hash(t.name)), str(t.label))
        else:
            g.node(str(hash(t.name)), '', style='filled', fillcolor='black')

    g.attr('node', shape='circle', fixedsize='true', width='0.75')
    for p in net.places:
        if highlight_marking and p in highlight_marking:
            g.node(str(hash(p.name)), 
                   str(highlight_marking[p]), 
                   style='filled', 
                   fillcolor=highlight_color,
                   xlabel=str(p.name))
        else:
            g.node(str(hash(p.name)), '', xlabel=str(p.name))

    for a in net.arcs:
        g.edge(str(hash(a.source.name)), str(hash(a.target.name)))

    g.attr(overlap='false')
    g.attr(fontsize='11')

    return g


def dot2bytes(g, format='png', renderer='dot', formatter='dot'):
    return g.pipe(format, renderer, formatter)


def add_state_to_rg(rg, state_label, prob_df, transcube_d_df_dict, top_k=5, color='transparent'):
    table_head = '<<TABLE BORDER="1" CELLBORDER="0" BGCOLOR="white">'
    table_tail = '</TABLE>>'
    label = '''<TR>
                    <TD colspan="4"><B>{state}</B></TD>
               </TR>
               <TR>
                    <TD><U>Activity</U></TD>
                    <TD><U>Likelihood</U></TD>
                    <TD><U>State</U></TD>
                    <TD><U>Likelihood</U></TD>
               </TR>'''.format(state=state_label)
    row_template = '''<TR><TD>{act}</TD><TD>{prob:.2f}%</TD>
                          <TD>{state}</TD><TD>{stateprob:.2f}%</TD></TR>'''
    i = 0
    for row in prob_df.itertuples(index=False):
        activity = row.activity
        prob = row.probability * 100.

        # get the state that the activity would most likely bring current state to 
        transmat_d = transcube_d_df_dict[activity]
        to_state = transmat_d.loc[state_label].idxmax()
        stateprob = transmat_d.loc[state_label].max() * 100.

        row = row_template.format(act=activity, prob=prob, 
                                  state=to_state, stateprob=stateprob)
        label += row

        i += 1
        if i >= top_k:
            break

    label = str(table_head + label + table_tail)
    # print('Adding node with label: {}'.format(label))
    rg.node(state_label, label=label, fillcolor=color, style='filled')


def get_adjacent_states(rg, highlight_state):
    adj_states = list()

    for trans in rg.transitions:
        if trans.from_state == highlight_state:
            adj_states.append(trans.to_state)
        elif trans.to_state == highlight_state:
            adj_states.append(trans.from_state)

    return adj_states


def rg2dot(rg, transcube_df_dict, transcube_d_df_dict,
           emitmat_df, emitmat_d_df, 
           highlight_marking_str=None, highlight_color='bisque', graph_attr=None):
    # set label as state name and assert that non-null highlight_marking_str exists
    highlight_state = None
    for state in rg.states:
        state.label = state.name
        if highlight_marking_str is not None and state.name == highlight_marking_str:
            highlight_state = state

    err_msg = 'Cannot find state with highlight_marking_str: {}'
    err_msg = err_msg.format(highlight_marking_str)
    if highlight_marking_str is not None and highlight_state is None:
        raise ValueError(err_msg)

    if highlight_marking_str is None:
        # show observation info of all nodes
        detailed_states = rg.states
    else:
        detailed_states = get_adjacent_states(rg, highlight_state)

    default_graph_attr = {
        'bgcolor': 'white',
        'rankdir': 'TB',
    }
    graph_attr = default_graph_attr if graph_attr is None else graph_attr
    g = Digraph(rg.name, engine='dot', graph_attr=graph_attr)

    g.attr('node')
    for state in rg.states:
        # observation likelihood
        obs_df = emitmat_d_df.loc[state.label].to_frame().reset_index(drop=False)
        obs_df.columns = ['activity', 'probability']
        obs_df.sort_values('probability', ascending=False, inplace=True)
        obs_df = obs_df.reset_index(drop=True)

        if state == highlight_state:
            add_state_to_rg(g, state.label, obs_df, transcube_d_df_dict, color=highlight_color)
        elif state in detailed_states:
            add_state_to_rg(g, state.label, obs_df, transcube_d_df_dict)
        else:
            # add as normal state without detaild info
            g.node(state.label, label=state.label)

    for trans in rg.transitions:
        g.edge(str(trans.from_state.name), str(trans.to_state.name), label=trans.name)

    # add_trans_to_g(g, transcube_d, state2int)

    g.attr(overlap='false')
    g.attr(fontsize='11')
    g.format = 'png'

    return g


def draw_undirected(G, node_map):
    pos = nx.spring_layout(G)
    
    # nodes
    node_color = 'lightblue'
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_color)

    # edges
    edges = [(u, v) for (u, v, d) in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    # create legend from node map
    handles = list()
    for key, val in node_map.items():
        patch = mpatches.Patch(color=node_color, label='{}:{}'.format(key, val))
        handles.append(patch)

    plt.legend(handles=handles)

