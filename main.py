# -*- coding: utf-8 -*-
"""
전국 시군구 고령화 / 유소년 지도 (스트림릿 앱)

이 앱이 하는 일 (초보자용 설명)
1) 인터넷에 있는 인구 데이터(csv.gz)와 지도 경계 데이터(geojson)를 내려받는다.
2) 슬라이더로 고른 "연도"의 데이터에서, 시군구별로 비율(65세 이상 또는 0~14세)을 계산한다.
   - 실제 데이터가 있는 연도(예: 2015~2026)는 그 해의 진짜 값을 계산한다.
   - 데이터가 없는 미래 연도(2027~2100)는, 그동안의 추세를 직선(1차 함수)으로
     늘려서 만든 "단순 예측값"을 보여준다. (진짜 인구 추계가 아니라 참고용이다!)
3) 예전 행정구역 코드(강원 42, 전북 45, 군위군 47720 등)를 요즘 코드로 바꿔서
   지도 경계 데이터와 최대한 잘 맞도록 손질한다. 그래도 안 맞는 지역은 회색으로 표시한다.
4) 지도 위에 전국 평균 / 최고 / 최저 지표를 카드로 보여준다.
5) 시도를 하나 골라서 그 지역만 확대해서 볼 수 있다.
6) 지도를 색칠하고, 그 아래 순위표(상위 10개 / 하위 10개)를 보여준다.
"""

import io
import gzip

import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------
# 0. 화면 기본 설정
# ---------------------------------------------------------
st.set_page_config(page_title="전국 고령화 지도", layout="wide")
st.title("🗺️ 전국 시군구별 고령화 · 유소년 지도")
st.caption("시군구 단위로 나이대별 인구 비율을 색칠해서 보여줍니다. (2027년 이후는 추세 예측치)")

# 데이터가 있는 인터넷 주소
POP_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/population_yearly.csv.gz"
GEO_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/boundaries/sigungu_kr.geojson"

# 예측 가능한 마지막 연도 (문제에서 요청한 값)
FORECAST_END_YEAR = 2100

# 회색으로 표시할 "코드가 안 맞는 지역" 항목 이름
NO_MATCH_LABEL = "코드 불일치(회색)"
NO_MATCH_COLOR = "#bdbdbd"

# 지표(고령화 / 유소년)별로 필요한 설정을 한곳에 모아둔다.
# - key: 내부에서 구분할 이름
# - age_start, age_end: 계산에 쓸 나이 범위 (양 끝 포함, 100세 이상은 elderly만 별도 처리)
# - fixed_edges/labels: 연도가 바뀌어도 구간 경계값이 그대로여야 하면 값을 넣어둔다.
#   (고령화율은 문제에서 정해준 19/23/28/38% 경계값을 그대로 쓴다)
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
# 1. 인구 데이터 / 지도 경계 데이터 내려받기
# ---------------------------------------------------------
@st.cache_data(show_spinner="인구 데이터를 불러오는 중입니다...")
def load_population():
    """인구 csv.gz 파일을 내려받아서 표(DataFrame)로 만든다."""
    response = requests.get(POP_URL)
    response.raise_for_status()

    # 압축을 풀어서 진짜 csv 내용으로 바꾼다.
    raw_bytes = gzip.decompress(response.content)

    # '코드'는 계산용 숫자가 아니라 이름표이므로, 무조건 글자(str)로 읽는다.
    df = pd.read_csv(io.BytesIO(raw_bytes), dtype={"코드": str})
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
    """
    실제 데이터가 있는 연도 하나를 받아서, 시군구별 "해당 나이 비율(%)" 표를 만든다.
    코드는 fix_old_code()로 보정한 뒤 합산한다.
    """
    df = pop_df[pop_df["연도"] == year].copy()

    # 코드 앞 5자리 = 시군구 코드. 그 다음 옛 코드를 새 코드로 보정한다.
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
    """실제 데이터가 있는 모든 연도의 결과를 하나의 표로 쌓아 올린다. (미래 예측의 재료가 된다)"""
    years = sorted(pop_df["연도"].unique())
    frames = [compute_sigungu_ratio(pop_df, y, age_start, age_end) for y in years]
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def forecast_ratio_for_year(history_long: pd.DataFrame, target_year: int):
    """
    실제 데이터가 없는 미래 연도의 비율을, 시군구별 과거 추세에 1차 직선(선형회귀)을
    맞춰서 예측한다. 과거 자료 점이 2개 미만인 지역은 마지막 실제값을 그대로 사용한다.
    예측값은 0~100% 범위를 벗어나지 않도록 잘라낸다.
    """
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

        # 이름과 인구 규모는 마지막 실제 연도 값을 그대로 가져다 쓴다.
        # (미래의 실제 인구 수는 알 수 없으므로, 비율을 계산할 때 가중치로만 활용한다)
        if code in last_year_df.index:
            시도 = last_year_df.loc[code, "시도"]
            시군구 = last_year_df.loc[code, "시군구"]
            전체인구 = last_year_df.loc[code, "전체인구"]
        else:
            시도 = group["시도"].iloc[-1]
            시군구 = group["시군구"].iloc[-1]
            전체인구 = group["전체인구"].iloc[-1]

        rows.append(
            {
                "시군구코드": code,
                "시도": 시도,
                "시군구": 시군구,
                "전체인구": 전체인구,
                "비율": round(predicted, 2),
            }
        )

    result = pd.DataFrame(rows)
    result["대상인구"] = result["전체인구"] * result["비율"] / 100
    result["연도"] = target_year
    return result


@st.cache_data(show_spinner=False)
def compute_youth_auto_edges(history_long: pd.DataFrame):
    """
    유소년 비율은 고령화율보다 훨씬 낮은 범위에 몰려 있어서, 같은 구간을 쓰면 지도가
    거의 한 색이 되어 버린다. 그래서 실제 데이터가 있는 모든 연도를 다 모아서, 값을
    다섯 덩어리로 나누는 경계값을 자동으로 계산한다. (20% / 40% / 60% / 80% 분위수)
    """
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
# 3. 데이터 불러오기
# ---------------------------------------------------------
population_df = load_population()
geojson_data = load_geojson()
geojson_codes = {feat["properties"]["코드"] for feat in geojson_data["features"]}

min_year = int(population_df["연도"].min())
max_actual_year = int(population_df["연도"].max())  # 실제 데이터가 있는 마지막 연도

# ---------------------------------------------------------
# 4. 화면 상단 컨트롤: 연도 슬라이더 / 지표 선택 / 시도 확대
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

# 선택한 지표의 연도별 실제 데이터(추세 예측의 재료 겸, 실제 연도 계산 결과)
history_long = build_history_long(population_df, metric_info["age_start"], metric_info["age_end"])

# 선택한 연도의 시군구별 비율표를 만든다. (실제 연도면 그대로, 미래 연도면 추세 예측)
if is_forecast:
    sigungu_df = forecast_ratio_for_year(history_long, selected_year)
else:
    sigungu_df = history_long[history_long["연도"] == selected_year].reset_index(drop=True)

# 구간 경계값 정하기: 고령화율은 고정값, 유소년 비율은 데이터 기반 자동 계산(실제 데이터로만 계산)
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
        "시군구별로 직선(선형회귀)으로 늘려서 만든 **단순 예측값**입니다. 실제 인구 추계 기관의 "
        "정교한 모델이 아니라는 점을 참고해 주세요."
    )

# ---------------------------------------------------------
# 5. 지표 카드 3장 (전국 평균 / 최고 / 최저)
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
# 6. 지도 그리기
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
    hover_data={
        "시도": True,
        "비율": ":.2f",
        "시군구코드": False,
        "구간": False,
    },
    labels={"구간": "구간", "비율": f"{metric_label}(%)", "시도": "시도"},
)

# 배경 지도 타일 없이, 선택된 지역의 경계선만 딱 맞춰서 보이게 설정한다.
fig.update_geos(fitbounds="locations", visible=False)
fig.update_traces(marker_line_color="white", marker_line_width=0.5)
fig.update_layout(
    margin={"r": 0, "t": 10, "l": 0, "b": 0},
    legend_title_text=f"{metric_label} 구간",
    height=700,
)

st.plotly_chart(fig, use_container_width=True)

# ---- 코드가 안 맞는 지역 안내 ----
mismatch_df = sigungu_df[sigungu_df["구간"] == NO_MATCH_LABEL]
if len(mismatch_df) > 0:
    names = ", ".join(mismatch_df["시도"] + " " + mismatch_df["시군구"])
    st.warning(
        f"⚠️ 다음 지역은 {selected_year}년 기준 코드가 지도 경계 데이터와 맞지 않아 회색으로 표시했어요: {names}"
    )
else:
    st.success("모든 시군구가 지도 경계 데이터와 정상적으로 연결되었어요.")

# ---------------------------------------------------------
# 7. 지도 아래 순위 표
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
    f"{max_actual_year}년 이후 값은 시군구별 실제 추세를 직선으로 늘린 단순 예측치입니다."
)
