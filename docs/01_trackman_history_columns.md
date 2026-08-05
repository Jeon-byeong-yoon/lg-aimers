# trackman_history.csv 컬럼 분석

> **총 30개 컬럼** | 각 행(row) = 야구 경기에서 투수가 던진 **투구 1개**

---

## 🏷️ 식별자 / 인덱스 (ID & Index)

| # | 컬럼명 | 타입 | 설명 |
|---|--------|------|------|
| 1 | `trackman_id` | int | 각 투구 레코드의 고유 ID (row identifier) |
| 6 | `trackman_game_id` | str | 경기 고유 ID (날짜-구장-경기번호, e.g. `20190329-Gocheok-1`) |

---

## 📅 경기 시간 정보 (Game Context - Time)

| # | 컬럼명 | 타입 | 설명 | 예시 값 |
|---|--------|------|------|---------|
| 2 | `season` | int | 시즌 연도 | `2019` |
| 3 | `game_date` | str | 경기 날짜 (MM/DD/YYYY) | `03/29/2019` |
| 4 | `game_month` | int | 경기 월 (1~12) | `3`, `4`, `5` |
| 5 | `game_dayofweek` | int | 요일 (0=월요일, 6=일요일) | `4`=금요일 |

---

## ⚾ 경기 상황 (In-Game Situation)

| # | 컬럼명 | 타입 | 설명 | 예시 값 |
|---|--------|------|------|---------|
| 7 | `pitch_no` | int | 해당 경기에서 몇 번째 투구인지 (경기 내 누적 투구 번호) | `113`, `53` |
| 8 | `inning` | int | 이닝 (1~9+) | `1`~`8` |
| 9 | `top_bottom` | str | 초/말 구분 | `Top` (초), `Bottom` (말) |
| 10 | `balls_before` | int | 이 투구 직전의 볼 카운트 (0~3) | `0`, `1`, `2`, `3` |
| 11 | `strikes_before` | int | 이 투구 직전의 스트라이크 카운트 (0~2) | `0`, `1`, `2` |
| 12 | `outs_before` | int | 이 투구 직전의 아웃 카운트 (0~2) | `0`, `1`, `2` |
| 13 | `pitch_of_pa` | int | 해당 타석(Plate Appearance)에서 몇 번째 투구인지 | `1`~`13` |

---

## 👤 선수 정보 (Player Info)

| # | 컬럼명 | 타입 | 설명 | 예시 값 |
|---|--------|------|------|---------|
| 14 | `pitcher_trackman_id` | int | 투수 고유 ID | `502010` |
| 15 | `batter_trackman_id` | int | 타자 고유 ID | `75151`, `76812` |
| 16 | `pitcher_hand` | str | 투수 투구 방향 (Left/Right) | `Right` |
| 17 | `batter_hand` | str | 타자 타격 방향 (Left/Right) | `Right`, `Left` |
| 18 | `pitcher_team` | str | 투수 팀 코드 | `KIW_HER` (키움 히어로즈) |
| 19 | `batter_team` | str | 타자 팀 코드 | `SK_WYV` (SK 와이번스) |

### 팀 코드 참고
| 코드 | 팀 |
|------|----|
| `KIW_HER` | 키움 히어로즈 |
| `SK_WYV` | SK 와이번스 |
| `DOO_BEA` | 두산 베어스 |
| `LG_TWI` | LG 트윈스 |
| `LOT_GIA` | 롯데 자이언츠 |
| `NC_DIN` | NC 다이노스 |
| `KIA_TIG` | KIA 타이거즈 |
| (등 KBO 10개 구단) | ... |

---

## 🎯 구종 분류 (Pitch Type)

| # | 컬럼명 | 타입 | 설명 | 예시 값 |
|---|--------|------|------|---------|
| 20 | `tagged_pitch_type` | str | **사람이 수동으로 태깅한 구종** | `Fastball`, `Slider`, `Curveball`, `Cutter`, `ChangeUp` |
| 21 | `auto_pitch_type` | str | **트랙맨 알고리즘이 자동 분류한 구종** | `Fastball`, `Slider`, `Curveball`, `Cutter`, `ChangeUp` |
| 22 | `pitch_type_group` | str | 구종을 크게 3~4그룹으로 묶은 것 | `fastball`, `breaking`, `offspeed`, `other` |

> 📌 `tagged` vs `auto` 차이: tagged는 스카우터/코치가 육안 판별, auto는 AI 자동 분류. 두 값이 다를 수 있음.

---

## 📐 투구 물리 지표 (Pitch Physics / TrackMan Metrics)

| # | 컬럼명 | 타입 | 단위 | 설명 |
|---|--------|------|------|------|
| 23 | `rel_speed` | float | km/h | **릴리스 속도** - 공을 손에서 놓는 순간의 속도 |
| 24 | `spin_rate` | float | rpm | **스핀 레이트** - 공의 분당 회전수 (높을수록 변화 크거나 뜨는 힘 강함) |
| 25 | `induced_vert_break` | float | cm | **유도 수직 변화량** - 중력 제외하고 스핀으로 인한 수직 이동량 (양수=상승, 음수=하강) |
| 26 | `horz_break` | float | cm | **수평 변화량** - 공이 좌우로 얼마나 꺾이는지 (투수 시점 기준) |
| 27 | `extension` | float | m | **익스텐션** - 투구판에서 공 릴리스 지점까지의 거리 (길수록 타자에게 더 가깝게 느껴짐) |
| 28 | `rel_height` | float | m | **릴리스 높이** - 공을 놓는 높이 (지면 기준) |
| 29 | `rel_side` | float | m | **릴리스 사이드** - 공을 놓는 좌우 위치 (홈플레이트 중심 기준) |
| 30 | `zone_speed` | float | km/h | **존 통과 속도** - 공이 홈플레이트 부근을 통과할 때의 속도 (릴리스 속도보다 느림) |

---

## 📊 컬럼 그룹 요약

```
trackman_history.csv (30 cols)
├── 🏷️ 식별자          : trackman_id, trackman_game_id
├── 📅 시간 정보        : season, game_date, game_month, game_dayofweek
├── ⚾ 경기 상황        : pitch_no, inning, top_bottom, balls_before,
│                         strikes_before, outs_before, pitch_of_pa
├── 👤 선수 정보        : pitcher/batter_trackman_id, pitcher/batter_hand,
│                         pitcher/batter_team
├── 🎯 구종 분류        : tagged_pitch_type, auto_pitch_type, pitch_type_group
└── 📐 투구 물리 지표   : rel_speed, spin_rate, induced_vert_break, horz_break,
                          extension, rel_height, rel_side, zone_speed
```

---

## 💡 모델링 관점에서 중요한 컬럼

| 중요도 | 컬럼 | 이유 |
|--------|------|------|
| ⭐⭐⭐ | `rel_speed`, `spin_rate` | 투구 위력의 핵심 지표 |
| ⭐⭐⭐ | `induced_vert_break`, `horz_break` | 구질(변화구 방향/크기) 결정 |
| ⭐⭐⭐ | `pitch_type_group` | 타깃 변수와 직접 연관 가능성 높음 |
| ⭐⭐ | `balls_before`, `strikes_before`, `outs_before` | 투구 전략 상황 |
| ⭐⭐ | `pitch_of_pa` | 타석 내 투구 패턴 |
| ⭐⭐ | `pitcher_hand`, `batter_hand` | 좌/우 투타 매치업 |
| ⭐ | `zone_speed` | `rel_speed`와 상관 높으나 보완적 정보 |
| ⭐ | `extension`, `rel_height`, `rel_side` | 릴리스 포인트 (투수 구폼 특성) |
