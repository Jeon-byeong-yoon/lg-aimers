"""
pitcher_agg.pkl 생성 스크립트
trackman_history.csv → 투수별 물리 지표 집계 → model/ 폴더에 저장
"""
import os, gc, pickle, time
import pandas as pd

HISTORY_PATH = r"공모전 dataset\open\data\trackman_history.csv"
OUT_PATH     = r"scripts\submission_v2\model\pitcher_agg.pkl"

print("trackman_history.csv 로딩 중... (약 1~2분)")
t0 = time.time()

TM_COLS = ['pitcher_trackman_id','season','pitch_type_group',
           'rel_speed','spin_rate','induced_vert_break','horz_break',
           'extension','rel_height','rel_side','zone_speed']

history = pd.read_csv(HISTORY_PATH, encoding='utf-8-sig', usecols=TM_COLS)
history = history[history['season'] <= 2024].copy()
print(f"  로드 완료: {history.shape}  ({time.time()-t0:.1f}s)")

print("투수별 집계 중...")
pitcher_agg = history.groupby('pitcher_trackman_id').agg(
    tm_speed_mean      =('rel_speed','mean'), tm_speed_std=('rel_speed','std'),
    tm_spin_mean       =('spin_rate','mean'), tm_spin_std=('spin_rate','std'),
    tm_vert_break_mean =('induced_vert_break','mean'), tm_vert_break_std=('induced_vert_break','std'),
    tm_horz_break_mean =('horz_break','mean'), tm_horz_break_std=('horz_break','std'),
    tm_rel_height_std  =('rel_height','std'), tm_rel_side_std=('rel_side','std'),
    tm_extension_mean  =('extension','mean'), tm_zone_speed_mean=('zone_speed','mean'),
    tm_pitch_count     =('rel_speed','count'),
).reset_index()
pitcher_agg['tm_speed_drop'] = pitcher_agg['tm_speed_mean'] - pitcher_agg['tm_zone_speed_mean']
pitcher_agg.drop(columns=['tm_zone_speed_mean'], inplace=True)

for grp in ['fastball','breaking','offspeed']:
    sub = history[history['pitch_type_group']==grp]
    pt = sub.groupby('pitcher_trackman_id').agg(
        **{f'tm_{grp}_speed_mean':('rel_speed','mean')},
        **{f'tm_{grp}_spin_mean':('spin_rate','mean')},
        **{f'tm_{grp}_rel_h_std':('rel_height','std')},
        **{f'tm_{grp}_rel_s_std':('rel_side','std')},
    ).reset_index()
    pitcher_agg = pitcher_agg.merge(pt, on='pitcher_trackman_id', how='left')

recent = history[history['season']==2024]
r_agg = recent.groupby('pitcher_trackman_id').agg(
    tm_recent_speed_mean=('rel_speed','mean'), tm_recent_spin_mean=('spin_rate','mean'),
    tm_recent_rel_h_std =('rel_height','std'),  tm_recent_rel_s_std=('rel_side','std'),
).reset_index()
pitcher_agg = pitcher_agg.merge(r_agg, on='pitcher_trackman_id', how='left')
pitcher_agg['tm_speed_trend']     = pitcher_agg['tm_recent_speed_mean'] - pitcher_agg['tm_speed_mean']
pitcher_agg['tm_release_trend_h'] = pitcher_agg['tm_recent_rel_h_std']  - pitcher_agg['tm_rel_height_std']
pitcher_agg['tm_release_trend_s'] = pitcher_agg['tm_recent_rel_s_std']  - pitcher_agg['tm_rel_side_std']
del history, recent, r_agg; gc.collect()

print(f"  집계 완료: {pitcher_agg.shape}  ({time.time()-t0:.1f}s)")

with open(OUT_PATH, 'wb') as f:
    pickle.dump(pitcher_agg, f)

print(f"\n✅ 저장 완료: {OUT_PATH}")
print(f"   투수 수: {len(pitcher_agg):,}명")
print(f"   피처 수: {pitcher_agg.shape[1]-1}개")
print(f"   총 소요: {time.time()-t0:.1f}s")
