"""Создать мини-сэмпл из исходных ga_* файлов.

Пример:
    python make_sample.py 20
(по умолчанию 20 session_id)
"""
import sys, pandas as pd, pathlib as pl

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
root = pl.Path(__file__).parent
sessions = pd.read_csv(root / "ga_sessions.csv", low_memory=False)
hits     = pd.read_csv(root / "ga_hits.csv",     low_memory=False)

keep_ids = sessions["session_id"].head(N).unique()
sess_sample = sessions[sessions["session_id"].isin(keep_ids)]
hits_sample = hits    [hits    ["session_id"].isin(keep_ids)]

root.joinpath("samples").mkdir(exist_ok=True)
sess_sample.to_csv(root / "samples" / "sessions_sample.csv", index=False)
hits_sample .to_csv(root / "samples" / "hits_sample.csv",     index=False)

print(f"Saved {len(sess_sample)} sessions / {len(hits_sample)} hits → samples/")
