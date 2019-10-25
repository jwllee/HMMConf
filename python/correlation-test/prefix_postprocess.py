import pandas as pd
import numpy as np
import time, os, argparse
import hmmconf


logger = hmmconf.utils.make_logger(__file__)


if __name__ == '__main__':
    info_msg = 'Postprocessing prefix alignment results...'
    logger.info(info_msg)

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store',
                        dest='result_dir',
                        help='Results directory of prefix alignment results')
    args = parser.parse_args()

    if args.result_dir is None:
        err_msg = 'Run as python ./prefix_postprocess.py -f [result_dir]'
        logger.error(err_msg)
        exit(0)

    outdir = os.path.join(args.result_dir, 'processed')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    for fname in os.listdir(args.result_dir):
        if not fname.endswith('.csv'):
            continue
        log = fname.replace('.csv', '')
        net = log.split('.pnml')[0].replace('log_', '')

        info_msg = 'Processing log {}'.format(log)
        logger.info(info_msg)

        fp = os.path.join(args.result_dir, fname)
        df = pd.read_csv(fp)

        info_msg = 'Log df: \n{}'.format(df.head())
        logger.info(info_msg)

        # keep only log, caseid, case_length, and cost
        cols = [
            'Length of the alignment found',
            'Log move cost of alignment',
            'Model move cost of alignment',
            'Synchronous move cost of alignment',
            'Cost of the alignment',
        ]
        not_empty = df['SP label'] != 'Empty'
        processed_df = df.loc[not_empty, cols].copy()
        split = df['SP label'].str.split('-', n=1, expand=True)
        processed_df['caseid'] = split[0]
        processed_df['case_length'] = split[1]
        processed_df['log'] = log
        processed_df['net'] = net

        cols = ['caseid', 'case_length', 'log', 'net'] + cols
        processed_df = processed_df[cols]

        out_fp = os.path.join(outdir, fname)
        processed_df.to_csv(out_fp, index=None)
