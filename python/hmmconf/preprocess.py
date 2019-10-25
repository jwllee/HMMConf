from copy import copy
from itertools import repeat


from pm4py.objects.petri import semantics
from . import pm_extra, example_data, utils


logger = utils.make_logger(__file__)


def token_pull(trans, net, marking, is_inv):
    """Perform token pull given a marking and a visible transition that we would like to execute. Algorithmically speaking,
    it corresponds to a BFS for a marking that enables the transition.

    :param trans: visible transition that we would like to execute
    :param net: petri net model
    :param marking: current marking
    :param is_inv: function that says if a transition is invisible
    :return: sequence of markings that leads to the enabling of transition or None if transition cannot be eventually enabled
    """
    debug_msg = 'Performing token pull at marking {} for target transition {}'
    debug_msg = debug_msg.format(marking, trans)
    logger.debug(debug_msg)

    def retrieve_marking_seq(node):
        debug_msg = 'Recursing from {} to get marking seq'
        debug_msg = debug_msg.format(node[0])
        logger.debug(debug_msg)

        marking_seq = list()
        while node[2] is not None: # parent is not the start marking
            marking_seq.insert(0, node[0])
            par_node = node[2]
            node = par_node
        return marking_seq

    enabled_trans = list(semantics.enabled_transitions(net, marking))
    enabled_trans = list(filter(is_inv, enabled_trans))
    start_node = (
        marking,                # current marking
        enabled_trans,          # enabled invisible transitions at marking
        None                    # parent node
    )
    queue = [start_node]
    visited = set()
    visited.add(marking)
    while queue:
        node = queue[0]
        cur_marking, enabled_trans, parent_node = node
        try:
            inv_tran = enabled_trans.pop(0)
            new_marking = semantics.execute(inv_tran, net, cur_marking)
            new_enabled_trans = list(semantics.enabled_transitions(net, new_marking))

            # check if the target transition is in the enabled transitions
            if trans in new_enabled_trans:
                # we are done since new_marking enables the target transitions
                # recurse back to get the sequence of markings that led to the transition to become enabled
                marking_seq = retrieve_marking_seq(node) + [new_marking]

                debug_msg = 'Recursed marking sequence: {}'
                debug_msg = debug_msg.format(marking_seq)
                logger.debug(debug_msg)

                return marking_seq

            # filter out all visible transitions and add to queue if there are enabled invisible transitions
            # also filter out transitions that lead to an already visited marking
            filtered_enabled = list()
            for t in new_enabled_trans:
                is_inv_t = is_inv(t)
                new_marking_t = semantics.execute(t, net, cur_marking)
                to_new_marking = new_marking_t not in visited
                if is_inv_t and to_new_marking:
                    filtered_enabled.append(t)
                    visited.add(to_new_marking)

            if len(filtered_enabled):
                # add new_marking to queue
                new_node = (
                    new_marking,
                    filtered_enabled,
                    node
                )
                queue.append(new_node)

        except:
            queue.pop(0)

    # cannot enable target transition
    return None


def token_replay_event(trans_list, net, marking, is_inv):
    """Performs token replay for a transition given the current marking.
    """
    debug_msg = 'Token replaying event {} at marking {}'
    debug_msg = debug_msg.format(trans_list[0].label, marking)
    logger.debug(debug_msg)

    marking_seq = list()
    enabled_trans = list(semantics.enabled_transitions(net, marking))
    target_trans = trans_list.pop(0)
    replayed = False
    while not replayed:
        if target_trans in enabled_trans:
            new_marking = semantics.execute(target_trans, net, marking)
            marking_seq.append(new_marking)
            break

        elif pm_extra.has_invisible(enabled_trans, is_inv):
            marking_seq_i = token_pull(target_trans, net, marking, is_inv)
            if marking_seq_i is not None:
                marking = marking_seq_i[-1]
                new_marking = semantics.execute(target_trans, net, marking)
                marking_seq_i.append(new_marking)
                marking_seq = marking_seq_i
                break

        # try next transition
        if trans_list:
            target_trans = trans_list.pop(0)
        # finished trying to replay transition
        else:
            marking_seq = None
            break

    return marking_seq


def get_marking_sequence(case, net, init_marking, final_marking, is_inv):
    """Performs token replay to get the marking sequence of a case through a petri net model, 
    only works for conforming cases, yields None otherwise.
    """
    cur_marking = copy(init_marking)
    marking_seq = []
    for event in case:
        assert isinstance(event, str), 'Event is assumed to be the activity label'
        # find the corresponding transition
        trans_list = pm_extra.find_transition(event, net.transitions)

        debug_msg = 'Mapped event {} to transitions: {}'
        debug_msg = debug_msg.format(event, trans_list)
        logger.debug(debug_msg)

        if len(trans_list) == 0:
            err_msg = 'Cannot map activity {} to any transition'.format(event)
            logger.error(err_msg)
            # log move
            return None

        marking_seq_i = token_replay_event(trans_list, net, cur_marking, is_inv)
        
        # break out of loop, the case is not perfectly conforming
        if marking_seq_i is None:
            debug_msg = 'Cannot continue replaying for event {} at marking {}'
            debug_msg = debug_msg.format(event, cur_marking)
            logger.debug(debug_msg)

            marking_seq = None
            return marking_seq

        # the current marking and all but the last marking from marking_seq_i is related
        # to observation of event
        last_marking = cur_marking
        cur_marking = marking_seq_i.pop(-1)
        
        n_repeats = 1 + len(marking_seq_i)
        repeated = repeat(event, n_repeats)
        markings = [last_marking] + marking_seq_i
        zipped = list(zip(repeated, markings))
        marking_seq = marking_seq + zipped

    # add the last marking with None
    marking_seq = marking_seq + [(None, cur_marking)]

    # raise a warning if cur_marking is not the final_marking but we have replayed everything
    if marking_seq and cur_marking != final_marking:
        err_msg = 'Current marking {} does not equal final marking {}'
        err_msg = err_msg.format(cur_marking, final_marking)
        # logger.error(err_msg)
        marking_seq = None

    return marking_seq

