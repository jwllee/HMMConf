import pandas as pd
import numpy as np


def transcube_to_dict(transcube, activity_list, state_list):
    df_dict = dict()
    for i in range(len(activity_list)):
        mat = transcube[i]
        act = activity_list[i]
        df = pd.DataFrame(mat, index=state_list, columns=state_list)
        df_dict[act] = df
    return df_dict


def emitmat_to_df(emitmat, activity_list, state_list):
    df = pd.DataFrame(emitmat, columns=activity_list, index=state_list)
    return df
