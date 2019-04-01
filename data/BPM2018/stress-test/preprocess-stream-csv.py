import pandas as pd
import os


if __name__ == '__main__':
    fp = os.path.join('.', 'stream.csv')
    df = pd.read_csv(fp)

    not_fake = df['activity_id'] != -1
    filtered_df = df.loc[not_fake,:]

    out_fp = os.path.join('.', 'filtered-stream.csv')
    filtered_df.to_csv(out_fp, index=None)
