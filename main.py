# -*- coding: utf-8 -*-
"""
전국 시군구 고령화 / 유소년 지도 + 생산가능인구 붕괴 타임어택 (스트림릿 앱)

이 앱이 하는 일 (초보자용 설명)
1) 인터넷에 있는 인구 데이터(csv.gz)와 지도 경계 데이터(geojson)를 내려받는다.
   - 용량을 줄이려고, 계산에 안 쓰는 '남_', '여_', '동' 열은 아예 읽지 않는다.
2) 맨 위에 "생산가능인구(15~64세) 1명이 아이·노인 1명을 부양하게 되는 시점"까지
   남은 시간을 1초 단위로 줄어드는 타이머와, 사람 모양으로 만든 모래시계로 보여준다.
3) 슬라이더로 고른 "연도"의 데이터에서, 시군구별로 비율(65세 이상 또는 0~14세)을 계산한다.
   - 실제 데이터가 있는 연도는 진짜 값을, 데이터가 없는 미래 연도(~2040)는 추세를
     직선으로 늘린 "단순 예측값"을 보여준다.
4) 예전 행정구역 코드(강원 42, 전북 45, 군위군 47720 등)를 요즘 코드로 바꿔서
   지도 경계 데이터와 최대한 잘 맞춘다. 그래도 안 맞는 지역은 회색으로 표시한다.
5) 지도 위에 전국 평균 / 최고 / 최저 지표를 카드로 보여준다.
6) 시도를 하나 골라서 그 지역만 확대해서 볼 수 있다.
7) 지도를 색칠하고, 그 아래 순위표(상위 10개 / 하위 10개)를 보여준다.
"""

import io
import gzip
import datetime

import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------
# 0. 화면 기본 설정
# ---------------------------------------------------------
st.set_page_config(page_title="전국 고령화 지도", layout="wide")
st.title("🗺️ 전국 시군구별 고령화 · 유소년 지도")
st.caption("시군구 단위로 나이대별 인구 비율을 색칠해서 보여줍니다. (2027년 이후는 추세 예측치)")

# 데이터가 있는 인터넷 주소
POP_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/population_yearly.csv.gz"
GEO_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/boundaries/sigungu_kr.geojson"

# 예측 가능한 마지막 연도 (데이터 용량 문제로 2100 -> 2040으로 줄임)
FORECAST_END_YEAR = 2040

# 회색으로 표시할 "코드가 안 맞는 지역" 항목 이름
NO_MATCH_LABEL = "코드 불일치(회색)"
NO_MATCH_COLOR = "#bdbdbd"

# 지표(고령화 / 유소년)별로 필요한 설정을 한곳에 모아둔다.
METRICS = {
    "고령화율 (65세 이상)": {
        "key": "elderly",
        "age_start": 65,
        "age_end": 100,       # 100세 이상까지 포함
        "fixed_edges": [-0.01, 19, 23, 28, 38, 100],
        "fixed_labels": ["19% 미만", "19%~23%", "23%~28%", "28%~38%", "38% 이상"],
    },
    "유소년 비율 (0~14세)": {
        "key": "youth",
        "age_start": 0,
        "age_end": 14,
        "fixed_edges": None,   # 아래에서 데이터 기반으로 자동 계산한다.
        "fixed_labels": None,
    },
}

# 구간 5단계에 쓸 색 (옅은 색 -> 진한 색 순서)
STEP_COLORS = ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"]


# ---------------------------------------------------------
# 1. 인구 데이터 / 지도 경계 데이터 내려받기 (용량을 줄여서 읽기)
# ---------------------------------------------------------
@st.cache_data(show_spinner="인구 데이터를 불러오는 중입니다...")
def load_population():
    """
    인구 csv.gz 파일을 내려받아서 표(DataFrame)로 만든다.
    실행이 무거워지지 않도록, 실제로 쓰는 열('연도','시도','시군구','코드','계_...')만 골라서 읽고,
    나이별 인구 숫자는 int32로 읽어서 메모리를 아낀다.
    """
    response = requests.get(POP_URL)
    response.raise_for_status()
    raw_bytes = gzip.decompress(response.content)

    # 1) 먼저 열 이름만 살짝 들여다본다 (0줄만 읽기).
    header_df = pd.read_csv(io.BytesIO(raw_bytes), nrows=0)
    all_columns = list(header_df.columns)

    # 2) 실제로 쓸 열만 남긴다: 기본 정보 열 + '계_'로 시작하는 나이별 인구 열
    #    ('남_', '여_'로 시작하는 열은 이 앱에서 안 쓰므로 아예 읽지 않는다 -> 용량 대폭 절약)
    keep_columns = [c for c in all_columns if c in ("연도", "시도", "시군구", "코드") or c.startswith("계_")]

    dtype_map = {"코드": str}
    for c in keep_columns:
        if c.startswith("계_"):
            dtype_map[c] = "int32"  # 인구 숫자는 int32면 충분해서 메모리를 절반 이하로 줄여준다.

    # 3) 골라둔 열만, 가벼운 타입으로 다시 읽는다.
    df = pd.read_csv(io.BytesIO(raw_bytes), usecols=keep_columns, dtype=dtype_map)
    return df


@st.cache_data(show_spinner="지도 경계 데이터를 불러오는 중입니다...")
def load_geojson():
    response = requests.get(GEO_URL)
    response.raise_for_status()
    return response.json()


def fix_old_code(code: str) -> str:
    """
    옛날 행정구역 코드를 요즘 지도 경계 데이터의 코드에 맞게 바꿔준다.
    - 강원(옛 42) -> 51
    - 전북(옛 45) -> 52
    - 군위군(옛 47720, 경북 소속) -> 27720 (대구 소속으로 변경)
    """
    if code == "47720":
        return "27720"
    if code.startswith("42"):
        return "51" + code[2:]
    if code.startswith("45"):
        return "52" + code[2:]
    return code


# ---------------------------------------------------------
# 2. 나이 열 이름 찾기 + 시군구별 비율 계산 (실제 데이터가 있는 연도용)
# ---------------------------------------------------------
def get_age_columns(pop_df: pd.DataFrame, age_start: int, age_end: int):
    """'계_숫자세' 형태의 열 중에서, 지정한 나이 범위(양 끝 포함)에 해당하는 열만 골라낸다."""
    cols = []
    for age in range(age_start, min(age_end, 99) + 1):
        col_name = f"계_{age}세"
        if col_name in pop_df.columns:
            cols.append(col_name)
    if age_end >= 100 and "계_100세 이상" in pop_df.columns:
        cols.append("계_100세 이상")
    return cols


def compute_sigungu_ratio(pop_df: pd.DataFrame, year: int, age_start: int, age_end: int):
    """실제 데이터가 있는 연도 하나를 받아서, 시군구별 "해당 나이 비율(%)" 표를 만든다."""
    df = pop_df[pop_df["연도"] == year].copy()
    df["시군구코드"] = df["코드"].str[:5].map(fix_old_code)

    total_age_cols = [c for c in df.columns if c.startswith("계_")]
    target_cols = get_age_columns(df, age_start, age_end)

    df["전체인구"] = df[total_age_cols].sum(axis=1)
    df["대상인구"] = df[target_cols].sum(axis=1)

    grouped = (
        df.groupby("시군구코드")
        .agg(
            시도=("시도", "first"),
            시군구=("시군구", "first"),
            전체인구=("전체인구", "sum"),
            대상인구=("대상인구", "sum"),
        )
        .reset_index()
    )
    grouped["비율"] = (grouped["대상인구"] / grouped["전체인구"] * 100).round(2)
    grouped["연도"] = year
    return grouped


@st.cache_data(show_spinner="연도별 실제 데이터를 준비하는 중입니다...")
def build_history_long(pop_df: pd.DataFrame, age_start: int, age_end: int):
    """실제 데이터가 있는 모든 연도의 결과를 하나의 표로 쌓아 올린다. (미래 예측의 재료)"""
    years = sorted(pop_df["연도"].unique())
    frames = [compute_sigungu_ratio(pop_df, y, age_start, age_end) for y in years]
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def forecast_ratio_for_year(history_long: pd.DataFrame, target_year: int):
    """미래 연도의 비율을, 시군구별 과거 추세에 1차 직선(선형회귀)을 맞춰서 예측한다."""
    last_year = history_long["연도"].max()
    last_year_df = history_long[history_long["연도"] == last_year].set_index("시군구코드")

    rows = []
    for code, group in history_long.groupby("시군구코드"):
        group = group.sort_values("연도")
        if len(group) >= 2:
            slope, intercept = np.polyfit(group["연도"], group["비율"], 1)
            predicted = slope * target_year + intercept
        else:
            predicted = group["비율"].iloc[-1]
        predicted = float(np.clip(predicted, 0, 100))

        if code in last_year_df.index:
            시도 = last_year_df.loc[code, "시도"]
            시군구 = last_year_df.loc[code, "시군구"]
            전체인구 = last_year_df.loc[code, "전체인구"]
        else:
            시도 = group["시도"].iloc[-1]
            시군구 = group["시군구"].iloc[-1]
            전체인구 = group["전체인구"].iloc[-1]

        rows.append(
            {"시군구코드": code, "시도": 시도, "시군구": 시군구, "전체인구": 전체인구, "비율": round(predicted, 2)}
        )

    result = pd.DataFrame(rows)
    result["대상인구"] = result["전체인구"] * result["비율"] / 100
    result["연도"] = target_year
    return result


@st.cache_data(show_spinner=False)
def compute_youth_auto_edges(history_long: pd.DataFrame):
    """유소년 비율은 값의 범위가 좁아서, 실제 데이터로 5단계 구간 경계값을 자동 계산한다."""
    ratios = history_long["비율"].dropna()
    cut_points = ratios.quantile([0.2, 0.4, 0.6, 0.8]).round(1).tolist()
    edges = [-0.01] + cut_points + [100]
    labels = [
        f"{cut_points[0]}% 미만",
        f"{cut_points[0]}%~{cut_points[1]}%",
        f"{cut_points[1]}%~{cut_points[2]}%",
        f"{cut_points[2]}%~{cut_points[3]}%",
        f"{cut_points[3]}% 이상",
    ]
    return edges, labels


def assign_bins(df: pd.DataFrame, edges, labels, matched_codes: set):
    """비율 값을 5단계 구간 글자로 바꾸고, geojson과 코드가 안 맞는 지역은 회색 구간으로 표시한다."""
    df = df.copy()
    binned = pd.cut(df["비율"], bins=edges, labels=labels)
    df["구간"] = binned.astype(str)
    df.loc[~df["시군구코드"].isin(matched_codes), "구간"] = NO_MATCH_LABEL
    return df


# ---------------------------------------------------------
# 3. 생산가능인구(15~64세) "붕괴 시점" 계산
#    -> 총부양비 {(유소년+고령)/생산가능인구 x 100} 가 100이 되는 시점
#       (생산가능인구 1명이 아이·노인 1명을 부양해야 하는 시점)
# ---------------------------------------------------------
@st.cache_data(show_spinner="전국 부양비 추세를 계산하는 중입니다...")
def compute_national_dependency_trend(pop_df: pd.DataFrame):
    years = sorted(pop_df["연도"].unique())
    rows = []
    for y in years:
        d = pop_df[pop_df["연도"] == y]
        youth_cols = get_age_columns(d, 0, 14)
        work_cols = get_age_columns(d, 15, 64)
        elder_cols = get_age_columns(d, 65, 100)

        youth = float(d[youth_cols].sum().sum())
        work = float(d[work_cols].sum().sum())
        elder = float(d[elder_cols].sum().sum())
        dependency_ratio = (youth + elder) / work * 100

        rows.append({"연도": y, "총부양비": dependency_ratio})
    return pd.DataFrame(rows)


def compute_collapse_target(dependency_trend: pd.DataFrame):
    """
    총부양비 추세를 직선(1차 함수)으로 늘려서, 100에 도달하는 "시점(날짜)"을 계산한다.
    추세가 증가하지 않으면(기울기<=0) None을 돌려준다.
    """
    slope, intercept = np.polyfit(dependency_trend["연도"], dependency_trend["총부양비"], 1)
    if slope <= 0:
        return None, slope, intercept

    crossing_year = (100 - intercept) / slope
    base_year = int(np.floor(crossing_year))
    frac = crossing_year - base_year
    target_date = datetime.datetime(base_year, 1, 1) + datetime.timedelta(days=frac * 365.25)
    return target_date, slope, intercept


# ---------------------------------------------------------
# 4. 사람 모양 모래시계 HTML 만들기
# ---------------------------------------------------------
def build_hourglass_rows_html():
    """
    위(아직 안 지나간 시간)와 아래(이미 지나간 시간)로 나뉜, 사람 모양(🧍) 모래시계를
    HTML로 만든다. 목(가운데 잘록한 부분)에서 가까운 줄부터 순서대로 id를 붙여서,
    자바스크립트가 시간에 따라 위에서 아래로 하나씩 "떨어지는" 것처럼 보이게 한다.
    """
    top_widths = [7, 5, 3, 1]      # 위쪽 줄 (넓은 줄 -> 목 쪽 좁은 줄)
    bottom_widths = [1, 3, 5, 7]   # 아래쪽 줄 (목 쪽 좁은 줄 -> 넓은 줄)

    # 목에서 가까운 순서대로 번호(id)를 매긴다.
    top_ids_by_row = {}
    counter = 0
    for row_idx in reversed(range(len(top_widths))):  # 목에 가장 가까운 줄(마지막 줄)부터
        width = top_widths[row_idx]
        top_ids_by_row[row_idx] = list(range(counter, counter + width))
        counter += width

    bottom_ids_by_row = {}
    counter = 0
    for row_idx in range(len(bottom_widths)):  # 목에 가장 가까운 줄(첫 줄)부터
        width = bottom_widths[row_idx]
        bottom_ids_by_row[row_idx] = list(range(counter, counter + width))
        counter += width

    rows_html = ""
    for row_idx in range(len(top_widths)):
        ids = top_ids_by_row[row_idx]
        cells = "".join(f'<span class="hg-person top" id="t{i}">🧍</span>' for i in ids)
        rows_html += f'<div class="hg-row">{cells}</div>\n'

    rows_html += '<div class="hg-neck"></div>\n'

    for row_idx in range(len(bottom_widths)):
        ids = bottom_ids_by_row[row_idx]
        cells = "".join(f'<span class="hg-person bottom" id="b{i}">🧍</span>' for i in ids)
        rows_html += f'<div class="hg-row">{cells}</div>\n'

    return rows_html


def build_timeattack_html(target_date, already_note: str, current_ratio_text: str):
    rows_html = build_hourglass_rows_html()

    already_passed = target_date is None
    target_iso = target_date.isoformat() if target_date else ""

    template = """
    <style>
      .ta-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 28px;
        align-items: center;
        justify-content: center;
        padding: 22px;
        border-radius: 18px;
        background: linear-gradient(135deg, #fff7ed, #ffe4d6);
        border: 1px solid #f3d5b5;
        margin-bottom: 6px;
        font-family: -apple-system, BlinkMacSystemFont, "Malgun Gothic", sans-serif;
      }
      .ta-timer-col { text-align: center; min-width: 260px; }
      .ta-label { font-size: 15px; color: #7c4a1e; margin-bottom: 8px; font-weight: 600; }
      .ta-timer {
        font-size: 32px; font-weight: 800; color: #b91c1c;
        font-variant-numeric: tabular-nums; letter-spacing: 0.5px;
      }
      .ta-status { font-size: 13px; color: #9a6a3a; margin-top: 8px; }
      .ta-sub { font-size: 12px; color: #a08060; margin-top: 4px; }

      .hg-col { display: flex; flex-direction: column; align-items: center; gap: 2px; }
      .hg-row { display: flex; gap: 3px; justify-content: center; }
      .hg-person {
        font-size: 15px; line-height: 1;
        transition: opacity .5s ease, transform .5s ease, filter .5s ease;
      }
      .hg-person.top { opacity: 1; filter: none; }
      .hg-person.top.drained { opacity: 0.12; transform: scale(0.6); filter: grayscale(1); }
      .hg-person.bottom { opacity: 0.12; transform: scale(0.6); filter: grayscale(1); }
      .hg-person.bottom.filled { opacity: 1; transform: scale(1); filter: none; }
      .hg-neck { height: 5px; width: 26px; background: #c98a4b; border-radius: 3px; margin: 3px 0; }
      .hg-caption { font-size: 11px; color: #a08060; margin-top: 6px; text-align: center; }
    </style>

    <div class="ta-wrap">
      <div class="ta-timer-col">
        <div class="ta-label">⏳ 생산가능인구(15~64세) 1명이<br>아이·노인 1명을 부양하게 되는 시점까지</div>
        <div class="ta-timer" id="ta-timer">계산 중...</div>
        <div class="ta-status" id="ta-status"></div>
        <div class="ta-sub">__ALREADY_NOTE__</div>
        <div class="ta-sub">__CURRENT_RATIO_TEXT__</div>
      </div>
      <div class="hg-col">
        __ROWS_HTML__
        <div class="hg-caption">사람 모양 모래시계 : 위=아직 남은 시간, 아래=이미 지나간 시간</div>
      </div>
    </div>

    <script>
    (function () {
      const alreadyPassed = __ALREADY_PASSED__;
      const targetIso = "__TARGET_ISO__";
      const timerEl = document.getElementById("ta-timer");
      const statusEl = document.getElementById("ta-status");
      const startTime = Date.now();

      function pad(n) { return String(n).padStart(2, "0"); }

      function setHourglass(frac) {
        const numDrained = Math.floor(Math.min(1, Math.max(0, frac)) * 16);
        for (let i = 0; i < 16; i++) {
          const t = document.getElementById("t" + i);
          const b = document.getElementById("b" + i);
          if (!t || !b) continue;
          if (i < numDrained) {
            t.classList.add("drained");
            b.classList.add("filled");
          } else {
            t.classList.remove("drained");
            b.classList.remove("filled");
          }
        }
      }

      if (alreadyPassed || !targetIso) {
        timerEl.textContent = "계산 불가";
        statusEl.textContent = "지금 추세로는 방향이 반대라 시점을 계산할 수 없어요.";
        setHourglass(0);
        return;
      }

      const target = new Date(targetIso).getTime();
      const totalMs = Math.max(target - startTime, 1);

      function tick() {
        const now = Date.now();
        const diff = target - now;

        if (diff <= 0) {
          const overMs = now - target;
          const overDays = Math.floor(overMs / 86400000);
          timerEl.textContent = "⏰ 이미 도달";
          statusEl.textContent = "추세 기준으로 " + overDays + "일 전에 이미 지났어요.";
          setHourglass(1);
          return;
        }

        const days = Math.floor(diff / 86400000);
        const hours = Math.floor((diff % 86400000) / 3600000);
        const mins = Math.floor((diff % 3600000) / 60000);
        const secs = Math.floor((diff % 60000) / 1000);

        timerEl.textContent = days + "일 " + pad(hours) + "시간 " + pad(mins) + "분 " + pad(secs) + "초";

        const frac = (now - startTime) / totalMs;
        setHourglass(frac);
      }

      tick();
      setInterval(tick, 1000);
    })();
    </script>
    """

    html = (
        template.replace("__ROWS_HTML__", rows_html)
        .replace("__ALREADY_PASSED__", "true" if already_passed else "false")
        .replace("__TARGET_ISO__", target_iso)
        .replace("__ALREADY_NOTE__", already_note)
        .replace("__CURRENT_RATIO_TEXT__", current_ratio_text)
    )
    return html


# ---------------------------------------------------------
# 5. 데이터 불러오기
# ---------------------------------------------------------
population_df = load_population()
geojson_data = load_geojson()
geojson_codes = {feat["properties"]["코드"] for feat in geojson_data["features"]}

min_year = int(population_df["연도"].min())
max_actual_year = int(population_df["연도"].max())  # 실제 데이터가 있는 마지막 연도

# ---------------------------------------------------------
# 6. 생산가능인구 붕괴 타임어택 (페이지 맨 위, 눈에 잘 띄게)
# ---------------------------------------------------------
dependency_trend = compute_national_dependency_trend(population_df)
collapse_target, dep_slope, dep_intercept = compute_collapse_target(dependency_trend)
latest_dependency_ratio = dependency_trend.sort_values("연도").iloc[-1]["총부양비"]

already_note = (
    f"({min_year}~{max_actual_year}년 실제 추세를 직선으로 늘린 단순 추정치이며, "
    "공식 인구 추계와 다를 수 있어요)"
)
current_ratio_text = f"현재(추세 기준) 총부양비 약 {latest_dependency_ratio:.1f}% (생산가능인구 100명당 부양인구 수)"

timeattack_html = build_timeattack_html(collapse_target, already_note, current_ratio_text)
components.html(timeattack_html, height=340, scrolling=False)

st.divider()

# ---------------------------------------------------------
# 7. 화면 상단 컨트롤: 연도 슬라이더 / 지표 선택 / 시도 확대
# ---------------------------------------------------------
control_col1, control_col2, control_col3 = st.columns([2, 2, 1.5])

with control_col1:
    selected_year = st.slider(
        f"연도 선택 (2015~{max_actual_year}은 실제 데이터, {max_actual_year + 1}~{FORECAST_END_YEAR}은 추세 예측)",
        min_value=min_year,
        max_value=FORECAST_END_YEAR,
        value=max_actual_year,
        step=1,
    )

with control_col2:
    metric_label = st.selectbox("지표 선택", list(METRICS.keys()))

metric_info = METRICS[metric_label]
is_forecast = selected_year > max_actual_year

history_long = build_history_long(population_df, metric_info["age_start"], metric_info["age_end"])

if is_forecast:
    sigungu_df = forecast_ratio_for_year(history_long, selected_year)
else:
    sigungu_df = history_long[history_long["연도"] == selected_year].reset_index(drop=True)

if metric_info["fixed_edges"] is not None:
    bin_edges = metric_info["fixed_edges"]
    bin_labels = metric_info["fixed_labels"]
else:
    bin_edges, bin_labels = compute_youth_auto_edges(history_long)

color_map = dict(zip(bin_labels, STEP_COLORS))
color_map[NO_MATCH_LABEL] = NO_MATCH_COLOR
category_orders = bin_labels + [NO_MATCH_LABEL]

sigungu_df = assign_bins(sigungu_df, bin_edges, bin_labels, geojson_codes)

with control_col3:
    sido_options = ["전국"] + sorted(sigungu_df["시도"].dropna().unique().tolist())
    selected_sido = st.selectbox("시도 확대 보기", sido_options)

if is_forecast:
    st.info(
        f"📈 {selected_year}년은 실제 조사 데이터가 없어요. {min_year}~{max_actual_year}년 추세를 "
        "시군구별로 직선(선형회귀)으로 늘려서 만든 **단순 예측값**입니다."
    )

# ---------------------------------------------------------
# 8. 지표 카드 3장 (전국 평균 / 최고 / 최저)
# ---------------------------------------------------------
valid_df = sigungu_df.dropna(subset=["비율"])

national_ratio = sigungu_df["대상인구"].sum() / sigungu_df["전체인구"].sum() * 100
top_row = valid_df.sort_values("비율", ascending=False).iloc[0]
bottom_row = valid_df.sort_values("비율", ascending=True).iloc[0]

card1, card2, card3 = st.columns(3)
card1.metric(f"전국 {metric_label}", f"{national_ratio:.2f}%")
card2.metric("가장 높은 시군구", f"{top_row['시군구']} ({top_row['시도']})", f"{top_row['비율']:.2f}%")
card3.metric("가장 낮은 시군구", f"{bottom_row['시군구']} ({bottom_row['시도']})", f"{bottom_row['비율']:.2f}%")

st.divider()

# ---------------------------------------------------------
# 9. 지도 그리기
# ---------------------------------------------------------
title_suffix = " (추세 예측)" if is_forecast else ""
st.subheader(f"{selected_year}년 기준 시군구별 {metric_label}{title_suffix}")

map_df = sigungu_df if selected_sido == "전국" else sigungu_df[sigungu_df["시도"] == selected_sido]

fig = px.choropleth(
    map_df,
    geojson=geojson_data,
    locations="시군구코드",
    featureidkey="properties.코드",
    color="구간",
    category_orders={"구간": category_orders},
    color_discrete_map=color_map,
    hover_name="시군구",
    hover_data={"시도": True, "비율": ":.2f", "시군구코드": False, "구간": False},
    labels={"구간": "구간", "비율": f"{metric_label}(%)", "시도": "시도"},
)

fig.update_geos(fitbounds="locations", visible=False)
fig.update_traces(marker_line_color="white", marker_line_width=0.5)
fig.update_layout(
    margin={"r": 0, "t": 10, "l": 0, "b": 0},
    legend_title_text=f"{metric_label} 구간",
    height=700,
)

st.plotly_chart(fig, use_container_width=True)

mismatch_df = sigungu_df[sigungu_df["구간"] == NO_MATCH_LABEL]
if len(mismatch_df) > 0:
    names = ", ".join(mismatch_df["시도"] + " " + mismatch_df["시군구"])
    st.warning(f"⚠️ 다음 지역은 {selected_year}년 기준 코드가 지도 경계 데이터와 맞지 않아 회색으로 표시했어요: {names}")
else:
    st.success("모든 시군구가 지도 경계 데이터와 정상적으로 연결되었어요.")

# ---------------------------------------------------------
# 10. 지도 아래 순위 표
# ---------------------------------------------------------
st.subheader(f"{metric_label} 순위{title_suffix}")

col_high, col_low = st.columns(2)

display_df = valid_df[["시도", "시군구", "비율"]].copy()
ratio_col = f"{metric_label}(%)"
display_df = display_df.rename(columns={"비율": ratio_col})

top10 = display_df.sort_values(ratio_col, ascending=False).head(10).reset_index(drop=True)
bottom10 = display_df.sort_values(ratio_col, ascending=True).head(10).reset_index(drop=True)
top10.index = top10.index + 1
bottom10.index = bottom10.index + 1

with col_high:
    st.markdown(f"**🔺 {metric_label} 높은 시군구 TOP 10**")
    st.dataframe(top10, use_container_width=True)

with col_low:
    st.markdown(f"**🔻 {metric_label} 낮은 시군구 TOP 10**")
    st.dataframe(bottom10, use_container_width=True)

st.caption(
    "※ 비율 = (해당 나이 인구 / 전체 인구) × 100. 시군구 코드(앞 5자리) 기준으로 "
    "읍·면·동 인구를 합산해서 계산했고, 옛 행정구역 코드는 최신 코드로 보정했습니다. "
    f"{max_actual_year}년 이후 값은 시군구별 실제 추세를 직선으로 늘린 단순 예측치입니다. "
    "상단의 '붕괴 타임어택'은 전국 총부양비 추세로 계산한 참고용 추정치입니다."
)
