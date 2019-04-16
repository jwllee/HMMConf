import pandas as pd
import os, sys
from pandas.api.types import CategoricalDtype


if __name__ == '__main__':
    outdir = os.path.join('.', 'processed')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    for fp in os.listdir('.'):
        if not os.path.isfile(fp):
            continue
        if fp.endswith('.py'):
            continue

        print('Processing {}'.format(fp))
        df = pd.read_csv(fp, sep='\t')
        print('df shape: {}'.format(df.shape))
        print('Head: \n{}'.format(df.head()))

        n_cases = df['T:concept:name'].unique().shape[0]
        case_length_stats = df[['T:concept:name', 'id']].groupby('T:concept:name').count().describe()
        print('No. of cases: {}'.format(n_cases))
        print('Case length stats: \n{}'.format(case_length_stats))

        # rename
        df.columns = ['caseid', 'activity', 'id']

        # add activity id
        ordered_acts = sorted(list(set(df['activity'])))
        activity_cat_type = CategoricalDtype(categories=ordered_acts, ordered=True)
        df['activity_id'] = df.activity.astype(activity_cat_type).cat.codes
        df = df[['id', 'caseid', 'activity', 'activity_id']]

        out_fp = os.path.join(outdir, fp)
        df.to_csv(out_fp, index=None)
