import glob, os, sys

import argparse
import pandas as pd
import numpy as np
from numpy import nan as NaN
import ast


RESPONSE_MAPPING = {
    "h_flip": {
        "v_flip": {
            "a": {
                "response_txt": "unseen",
                "response_confidence": True,
                "response": -2.0,
            },
            "b": {
                "response_txt": "seen",
                "response_confidence": True,
                "response": 2.0,
            },
            "x": {
                "response_txt": "unseen",
                "response_confidence": False,
                "response": -1.0,
            },
            "y": {
                "response_txt": "seen",
                "response_confidence": False,
                "response": 1.0,
            },
        },
        "v_noflip": {
            "a": {
                "response_txt": "unseen",
                "response_confidence": True,
                "response": -2.0,
            },
            "b": {
                "response_txt": "unseen",
                "response_confidence": False,
                "response": -1.0,
            },
            "x": {
                "response_txt": "seen",
                "response_confidence": True,
                "response": 2.0,
            },
            "y": {
                "response_txt": "seen",
                "response_confidence": False,
                "response": 1.0,
            },
        },
    },
    "h_noflip": {
        "v_flip": {
            "a": {
                "response_txt": "seen",
                "response_confidence": False,
                "response": 1.0,
            },
            "b": {
                "response_txt": "seen",
                "response_confidence": True,
                "response": 2.0,
            },
            "x": {
                "response_txt": "unseen",
                "response_confidence": False,
                "response": -1.0,
            },
            "y": {
                "response_txt": "unseen",
                "response_confidence": True,
                "response": -2.0,
            },
        },
        "v_noflip": {
            "a": {
                "response_txt": "seen",
                "response_confidence": False,
                "response": 1.0,
            },
            "b": {
                "response_txt": "unseen",
                "response_confidence": False,
                "response": -1.0,
            },
            "x": {
                "response_txt": "seen",
                "response_confidence": True,
                "response": 2.0,
            },
            "y": {
                "response_txt": "unseen",
                "response_confidence": True,
                "response": -2.0,
            },
        },
    },
}


def multiple_kp(row):
    '''
    Flags trials with multiple reccorded key presses
    Returns bool
    '''
    if pd.isnull(row["response_txt"]):
        return False
    else:
        keypresses = ast.literal_eval(row["keypresses"])
        return len(keypresses) > 1


def get_kp_val(row, column="response_txt", get_rt=False, first_kp=True):
    '''
    Derives response value from logged keypress
    Args:
        column (str): choices = ["response", "response_txt", "response_confidence"]
        get_rt (bool): if True, return keypress reaction time
        first_kp (bool): if True, returns val from first key press,
        else last key press
    '''
    if pd.isnull(row["response_txt"]):
        return NaN
    else:
        keypresses = ast.literal_eval(row["keypresses"])
        [k, t] = keypresses[0] if first_kp else keypresses[-1]

        if get_rt:
            return t - row["onset"]

        h_flip = "h_flip" if row["response_mapping_flip_h"] else "h_noflip"
        v_flip = "v_flip" if row["response_mapping_flip_v"] else "v_noflip"

        return RESPONSE_MAPPING[h_flip][v_flip][k][column]


def get_error(row, first_kp=True):
    '''
    Derives response accuracy (error = True/False) from logged keypress
    and condition (seen, unseen)
    Args:
        first_kp (bool): if True, returns error from first key press,
        else last key press
    '''
    if pd.isnull(row["response_txt"]):
        return NaN
    else:
        keypresses = ast.literal_eval(row["keypresses"])
        [k, t] = keypresses[0] if first_kp else keypresses[-1]

        h_flip = "h_flip" if row["response_mapping_flip_h"] else "h_noflip"
        v_flip = "v_flip" if row["response_mapping_flip_v"] else "v_noflip"

        kp_resp = RESPONSE_MAPPING[h_flip][v_flip][k]["response_txt"]

        return kp_resp != row["condition"]


def validate_keypresses(df):
    """
    Check that key press was properly processed in terms of response
    (text and num), error (accuracy) and RT

    args:
        df(pd.DataFrame): input run's *events.tsv loaded in pandas
    out:
        test_score (np.array[bool]): array of bool where True -> validated,
            False -> logged error in keypress processing
    """
    test_txt = (df["response_txt"] == df.apply(lambda row: get_kp_val(row), axis=1)).to_numpy()
    test_resp = (df["response"] == df.apply(lambda row: get_kp_val(row, column="response"), axis=1)).to_numpy()
    test_conf = (df["response_confidence"] == df.apply(lambda row: get_kp_val(row, column="response_confidence"), axis=1)).to_numpy()
    test_rt = np.isclose(df["response_time"],  df.apply(lambda row: get_kp_val(row, get_rt=True), axis=1))
    test_err = (df["error"] == df.apply(lambda row: get_error(row), axis=1)).to_numpy()

    test_score = test_txt * test_resp * test_conf * test_rt * test_err

    # do not flag NaN responses as erroneous entries
    test_score[df["response_txt"].isna()] = True

    return test_score


def update_kp_val(row, column="response_txt", get_rt=False):
    '''
    Derives behav metric response value from last logged keypress
    (reaction time, response (num and text), and response confidence)
    Args:
        column (str): choices = ["response", "response_txt", "response_confidence"]
        get_rt (bool): if True, return keypress reaction time
    '''
    if row["multiple_keypresses"]:
        if column=="error":
            return get_error(row, first_kp=False)
        else:
            return get_kp_val(row, column=column, get_rt=get_rt, first_kp=False)
    else:
        return row[column]


def add_lastkp_entries(df):
    """
    Inserts additional columns to process behav performance metrics based on
    last rather than first loggeg key press

    args:
        df (pd.DataFrame): input run's *events.tsv loaded in pandas
    out:
        df (pd.DataFrame): input df with added columns
    """
    for col, idx, rt in [
        ["response", 28, False], ["response_txt", 29, False],
        ["error", 30, False], ["response_confidence", 31, False],
        ["response_time", 32, True],
    ]:
        col_vals = df.apply(lambda row: update_kp_val(
            row, column=col, get_rt=rt), axis=1
        )
        df.insert(
            idx, f"{col}_lastkeypress", col_vals, allow_duplicates=True
        )

    return df


def insert_LBP_metrics(data_dir):
    """
    Inserts metrics that consider the subject's last button press
    rather than the first button press (which is the default)
    for analyses of behavioural performances on the THINGS memory task.
    """
    file_list = sorted(glob.glob(f"{data_dir}/sub-0*/ses*/func/*events.tsv"))

    for f in file_list:
        df = pd.read_csv(f, sep = "\t")

        # Validate logged behav metrics derived from first keypress
        valid_kp = validate_keypresses(df)
        assert np.sum(~valid_kp) == 0

        # Add multiple_keypresses column (bool)
        mkp = df.apply(lambda row: multiple_kp(row), axis=1)
        df.insert(27, "multiple_keypresses", mkp, allow_duplicates=True)
        # Add behav metrics derived from last keypress
        df = add_lastkp_entries(df)

        # save but do not overright for now
        df.to_csv(
            f"{f.replace('_events', '_desc-lkp_events')}",
            sep="\t", header=True, index=False,
        )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--idir',
        type=str,
        required=True,
        help='path to things.raw bids folder with *events.tsv output files',
    )
    args = parser.parse_args()

    insert_LBP_metrics(args.idir)


if __name__ == '__main__':
    sys.exit(main())
