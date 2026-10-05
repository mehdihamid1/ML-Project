import pandas as pd
from sklearn.model_selection import train_test_split
from .models import EXCLUDED


def prepare_data(frame):
    required = {'Label', 'SHA1', 'FirstSeenDate', 'Identify', 'ImportedDlls', 'ImportedSymbols'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(frame.columns))}')
    if frame['Label'].isna().any() or not set(frame['Label'].unique()).issubset({0, 1}):
        raise ValueError('Label must contain only 0 and 1')
    if frame['SHA1'].isna().any():
        raise ValueError('SHA1 must be present for duplicate control')
    conflicts = frame.groupby('SHA1')['Label'].nunique()
    conflicts = conflicts[conflicts > 1].index
    clean = frame[~frame['SHA1'].isin(conflicts)].drop_duplicates('SHA1').copy()
    if clean['Label'].nunique() != 2:
        raise ValueError('Training requires both classes')
    stats = {'raw_rows': len(frame), 'conflicting_hashes_removed': len(conflicts), 'clean_rows': len(clean), 'removed_rows': len(frame) - len(clean)}
    return clean, stats


def split_data(frame):
    return train_test_split(frame, test_size=0.2, stratify=frame['Label'], random_state=42)


def feature_frame(frame):
    return frame.drop(columns=EXCLUDED, errors='ignore')
