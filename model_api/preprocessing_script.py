"""
Convert raw GA logs ➜ model-ready features
──────────────────────────────────────────
If you already have a CSV that looks like `dataset.csv`, call
    df = enrich_for_ml(pd.read_csv("dataset.csv"))
Otherwise call
    df = build_dataset("data/ga_sessions.csv", "data/ga_hits.csv")
and feed the result to enrich_for_ml().
"""

from __future__ import annotations
import json, math, pickle
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------------------------
# 1. HELPERS ────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------------------

_TARGET_ACTIONS = {
    'sub_car_claim_submit_click','sub_car_request_submit_click','sub_callback_submit_click',
    'sub_open_dialog_click','start_chat','open_offer','form_request_call_sent',
    'callback requested','phone_entered','sub_submit_success','chat requested',
    'client initiate chat','click_on_request_call','show_phone_input',
    'name_entered','phone_entered_on_form_request_call'
}
_DEVICE_BRANDS  = ['apple', 'samsung', 'xiaomi', 'huawei']
_BROWSERS_TOP8  = ['chrome','safari','yandex','firefox','opera','edge','android','uc browser']
_DESKTOP_TAG    = 'desktop'
_RANDOM_STATE   = 42

_COMBOS_FILE = Path("models/valuable_combos.json")   # persisted after first train


# ────────────────────────── basic munging utils
def _fill_utm_source(df: pd.DataFrame) -> pd.Series:
    mapping = (df.dropna(subset=['utm_source'])
                 .groupby('client_id')['utm_source'].first().to_dict())
    return df['utm_source'].fillna(df['client_id'].map(mapping))


def _gcd_ratio(res: str) -> str|None:
    try:
        w,h = map(int, res.split('x'))
        g   = math.gcd(w, h)
        return f"{w//g}:{h//g}"
    except Exception:
        return None


def _top_brand(lst: list[str]) -> str:
    if not lst: return 'none'
    return Counter(lst).most_common(1)[0][0]


def _handle_lognorm_outliers(s: pd.Series, thr: float = 3) -> pd.Series:
    """robust-zscore, корректно работает на нулях"""
    log   = np.log1p(s)                         # zeros → 0
    med   = np.median(log)
    mad   = np.median(np.abs(log - med)) or 1   # fallback 1
    z_r   = np.abs((log - med) / (1.4826 * mad))
    return s.mask(z_r > thr, s.median())


# ------------------------------------------------------------------------------
# 2. GA SESSIONS / HITS CLEANUP  (exactly as in 01_Initial_analysis) ────────────
# ------------------------------------------------------------------------------

def _clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(['utm_keyword','device_os','device_model'], axis=1, errors='ignore')

    df['utm_source'] = _fill_utm_source(df)
    df = df.query("utm_medium != '(not set)'")
    df = df.query("device_screen_resolution not in ['(not set)','0x0']")

    # manual “banner” fix
    m = (df['utm_source']=='fDLlAcSmythWSCVMvqvL') & \
        (df['utm_adcontent']=='JNHcPlZPxEMWDnRiyoBf') & \
        (df['utm_campaign']=='LTuZkdKfxRGVceoWkVyg')
    df.loc[m, 'utm_medium'] = 'banner'

    df = df.drop(['utm_campaign','utm_adcontent'], axis=1, errors='ignore')

    df['visit_date']  = pd.to_datetime(df['visit_date'],   errors='coerce')
    df['visit_time']  = pd.to_datetime(df['visit_time'],   errors='coerce').dt.time
    
    if 'device_screen_resolution' in df.columns:
        # делаем строкой и безопасно сплитим "ШхВ"
        res = (
            df['device_screen_resolution']
              .astype(str)
              .str.lower()
              .str.replace('[^0-9x]', '', regex=True)        # чистим мусор
              .str.split('x', n=1, expand=True)
        )
        wh = pd.to_numeric(res[0], errors='coerce')
        hh = pd.to_numeric(res[1], errors='coerce')
        df['screen_area'] = wh * hh
    else:
        df['screen_area'] = np.nan

    # если всё равно NaN – подменяем медианой, чтобы модель не рушилась
    if df['screen_area'].isna().all():
        df['screen_area'] = 0
    else:
        df['screen_area'].fillna(df['screen_area'].median(), inplace=True)
        
    df['aspect_ratio']= df['device_screen_resolution'].map(_gcd_ratio)
    df  = df.drop(['device_screen_resolution'], axis=1)

    df['device_brand'] = df['device_brand'].fillna('unknown')

    for col in ['utm_medium','device_category','device_brand',
                'device_browser','geo_country','geo_city']:
        df[col] = df[col].str.lower()

    return df.dropna()


def _clean_hits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(['hit_time','event_value','hit_referer','event_label','hit_type'], axis=1)

    df['car_brand'] = df['hit_page_path'].str.extract(r'/cars/all/([^/?]+)').fillna('unknown')
    df  = df.drop('hit_page_path', axis=1)

    df['target'] = df['event_action'].isin(_TARGET_ACTIONS)

    agg = (df.groupby('session_id')
             .agg(event_categories_number = ('event_category','size'),
                  hit_number_median        = ('hit_number','median'),
                  car_brands_list          = ('car_brand', list))
             .reset_index())

    agg['car_brands_list'] = agg['car_brands_list'].apply(
        lambda lst:[b for b in lst if b!='unknown'])
    agg['top_car_brand']   = agg['car_brands_list'].apply(_top_brand)
    agg = agg.drop('car_brands_list', axis=1)

    agg = agg.merge(df.groupby('session_id')['target'].any(),
                    on='session_id', how='left')
    return agg


def build_dataset(sess_path:str|Path, hits_path:str|Path) -> pd.DataFrame:
    return (_clean_hits(pd.read_csv(hits_path))
              .merge(_clean_sessions(pd.read_csv(sess_path)),
                     on='session_id')
              .drop(columns=['session_id'])
              .dropna())
# ------------------------------------------------------------------------------
# 3.  EDA-LEVEL FEATURE ENGINEERING  (exact copy of EDA notebook) ──────────────
# ------------------------------------------------------------------------------

def _map_visit_number(v:int) -> int:
    return v if v < 4 else 4


def _map_top_car_brand(br:str,
                       top15: set[str],
                       down2: set[str]) -> int:
    if br in top15:      return 1
    if br in down2:      return 0
    return 2


def _recode_topN(val:str, top:list[str], rest_code:int) -> int:
    try: return top.index(val)
    except ValueError: return rest_code


def _generate_combos(df: pd.DataFrame,
                     cat_cols: List[str],
                     y: pd.Series,
                     min_target_share=.07,
                     min_size_share=.1) -> List[tuple[str,str,str]]:
    """Return list of (f1,f2,combo_value) triples that pass thresholds."""
    combos=[]
    total=len(df)
    for f1,f2 in combinations(cat_cols,2):
        tmp=(df[f1].astype(str)+'_'+df[f2].astype(str))
        stats=df.groupby(tmp)[y.name].agg(['size','mean'])
        stats=stats[(stats['mean']>min_target_share)&
                    (stats['size']>min_size_share*total)]
        for comb in stats.index:
            combos.append((f1,f2,comb))
    return combos


def enrich_for_ml(base_df: pd.DataFrame,
                  store_combos: bool = True) -> pd.DataFrame:
    """Apply every transformation done in the EDA notebook."""
    df = base_df.copy()

    # 0.  type / columns housekeeping
    df['target'] = df['target'].astype(int)
    df = df.drop('client_id', axis=1, errors='ignore')

    # 1.  visit_number recode
    df['visit_number'] = df['visit_number'].apply(_map_visit_number)

    # 2.  drop screen_area
    df = df.drop('screen_area', axis=1, errors='ignore')

    # 3.  date / time → engineered parts
    df['visit_date']  = pd.to_datetime(df['visit_date'])
    df['month']       = df['visit_date'].dt.month
    df['day_of_week'] = df['visit_date'].dt.dayofweek
    df['visit_hour']  = df['visit_time'].apply(lambda t: t.hour if pd.notna(t) else np.nan)
    _time_map = {**{h:'night'   for h in range(0,6)},
                 **{h:'morning' for h in range(6,11)},
                 **{h:'day'     for h in range(11,18)},
                 **{h:'evening' for h in range(18,24)}}
    df['time_of_day'] = df['visit_hour'].map(_time_map)
    df = df.drop(['visit_date','visit_time','visit_hour'], axis=1)

    # 4.  top_car_brand recode
    top15 = df['top_car_brand'].value_counts()[:15].index
    df['top_car_brand'] = df['top_car_brand'].apply(
        _map_top_car_brand, args=(set(top15),{'infiniti','honda'}))

    # 5.  utm_source / utm_medium recodes
    top5_source = df['utm_source'].value_counts()[:5].index.tolist()
    df['utm_source']  = df['utm_source'].apply(_recode_topN,
                                               args=(top5_source,5)).astype(int)

    top8_medium = df['utm_medium'].value_counts()[:8].index.tolist()
    df['utm_medium']  = df['utm_medium'].apply(_recode_topN,
                                               args=(top8_medium,8)).astype(int)

    # 6.  device_* recodes
    df['device_category'] = df['device_category'].apply(
        lambda x:x if x==_DESKTOP_TAG else 'mobile_device')
    df['device_brand']    = df['device_brand'].apply(
        lambda x:x if x in _DEVICE_BRANDS else 'other')

    _browser_map = {'safari (in-app)':'safari',
                    'mozilla':'firefox',
                    'android webview':'android',
                    'android runtime':'android',
                    'android browser':'android'}
    df['device_browser'] = df['device_browser'].replace(_browser_map)
    df['device_browser'] = df['device_browser'].apply(
        lambda x: x if x in _BROWSERS_TOP8 else 'other')

    # 7. aspect_ratio -> float -> drop
    def _norm_ratio(r):
        try: a,b = map(int,r.split(':')); return min(a,b)/max(a,b)
        except:  return np.nan
    df['aspect_ratio'] = df['aspect_ratio'].apply(_norm_ratio)
    df = df.drop('aspect_ratio', axis=1)

    # 8. geography
    df['geo_country'] = df['geo_country'].apply(
        lambda x: 'russia' if x=='russia' else 'no_russia')
    df['geo_city']    = df['geo_city'].apply(
        lambda x: x if x in ['moscow','saint petersburg'] else 'other')

    # 9. log-normal outliers
    for col in ['event_categories_number','hit_number_median']:
        df[col] = _handle_lognorm_outliers(df[col])

    # 10. binary combo features
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if _COMBOS_FILE.exists():
        combos = json.loads(_COMBOS_FILE.read_text())
    else:
        combos = _generate_combos(df, cat_cols, df['target'])
        if store_combos:
            _COMBOS_FILE.write_text(json.dumps(combos, indent=2))

    for f1,f2,combo in combos:
        new_col = f"{f1}_{f2}_{combo.replace(':','_').replace(' ','_')}"
        df[new_col] = ((df[f1].astype(str)+'_'+df[f2].astype(str)) == combo).astype(int)

    return df


# ------------------------------------------------------------------------------
# 4. ONE entry point for API
# ------------------------------------------------------------------------------

def preprocess_from_raw(sess_csv:str|Path, hits_csv:str|Path) -> pd.DataFrame:
    """Full chain: raw logs ➜ ML-ready DF (without target)."""
    df = build_dataset(sess_csv, hits_csv)
    df = enrich_for_ml(df, store_combos=False)
    return df.drop('target', axis=1, errors='ignore')
