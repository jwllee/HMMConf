import lxml, os, time
import numpy as np
import pandas as pd
from lxml import etree
from pandas.api.types import CategoricalDtype
from datetime import datetime


if __name__ == '__main__':
    fp = os.path.join('.', 'stream')

    # need to modify the stream file by wrapping the entire document in <items></items> tags
    modify_fp = os.path.join('.', 'stream-modified')
    with open(modify_fp, mode='w') as f:
        print('<items>', file=f)
        with open(fp) as f1:
            for line in f1.readlines():
                print(line, file=f)
        print('</items>', file=f)

    XTRACE_TAG = 'org.deckfour.xes.model.impl.XTraceImpl'
    LOG_TAG = 'log'
    TRACE_TAG = 'trace'
    EVENT_TAG = 'event'

    LOG_IND = 0 # the log element is xtrace element's first child
    TRACE_IND = 0 # the trace element is log element's first child
    CASEID_IND = 0 # the caseid element is trace element's first child
    EVENT_IND = 1
    ACTIVITY_IND = 1
    TIMESTAMP_IND = -1

    tag_err_msg = 'Expected element tag: {} is different to parsed: {}'

    events = list()

    start = time.time()
    with open(modify_fp, 'rb') as f:
        context = etree.iterparse(f, events=('end',), tag = XTRACE_TAG)

        i = 0
        for event, xtrace in context:
            # print('{}: {}'.format(event, xtrace.tag))

            log_elem = xtrace[LOG_IND]
            trace_elem = log_elem[TRACE_IND]
            event_elem = trace_elem[EVENT_IND]

            assert xtrace.tag.endswith(XTRACE_TAG), tag_err_msg.format(XTRACE_TAG, xtrace.tag)
            assert log_elem.tag.endswith(LOG_TAG), tag_err_msg.format(LOG_TAG, log_elem.tag)
            assert trace_elem.tag.endswith(TRACE_TAG), tag_err_msg.format(TRACE_TAG, trace_elem.tag)
            assert event_elem.tag.endswith(EVENT_TAG), tag_err_msg.format(EVENT_TAG, event_elem.tag)

            child_elems = {c.get('key'):c for c in event_elem}

            caseid_elem = trace_elem[CASEID_IND]
            activity_elem = child_elems['concept:name']
            timestamp_elem = child_elems['time:timestamp']

            assert caseid_elem.tag.endswith('string'), tag_err_msg.format('string', caseid_elem.tag)
            assert activity_elem.tag.endswith('string'), tag_err_msg.format('string', activity_elem.tag)
            assert timestamp_elem.tag.endswith('date'), tag_err_msg.format('date', timestamp_elem.tag)

            caseid = caseid_elem.get('value')
            activity = activity_elem.get('value')
            timestamp = datetime.fromisoformat(timestamp_elem.get('value'))

            events.append((caseid, activity, timestamp))

#            if i > 10:
#                break
#            i += 1
            xtrace.clear()
    end = time.time()
    print('Took {:.2f}s'.format(end - start))

    columns = ('caseid', 'activity', 'timestamp')
    df = pd.DataFrame(events, columns=columns)

    # put activity into categories
    uniq_activities = list(set(df['activity']))
    real_activities = sorted(list(filter(lambda a: a.startswith('Activity '), uniq_activities)))
    # fake_activities = sorted(list(filter(lambda a: not a.startswith('Activity '), uniq_activities)))
    # non-real activities are mapped to -1
    ordered_activities = real_activities

    print('Number of activities: {}'.format(len(ordered_activities)))
    print('Activities: {}'.format(ordered_activities))

    activity_cat_type = CategoricalDtype(categories=ordered_activities, ordered=True)
    df['activity_id'] = df.activity.astype(activity_cat_type).cat.codes
    df = df[['caseid', 'activity', 'activity_id', 'timestamp']]

    print('Dataframe shape: {}'.format(df.shape))
    print(df.head())

    out_fp = os.path.join('.', 'stream.csv')
    df.to_csv(out_fp, index=None)
