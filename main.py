# -*- coding: utf-8 -*-
"""
전국 시군구 고령화 지도 (스트림릿 앱)

이 앱이 하는 일 (초보자용 설명)
1) 인터넷에 있는 인구 데이터(csv.gz)와 지도 경계 데이터(geojson)를 내려받는다.
2) 인구 데이터에서 가장 최신 연도만 골라서, 시군구별로 "65세 이상 인구 비율(고령화율)"을 계산한다.
3) 지도 경계 데이터와 "코드"(행정구역 코드)를 기준으로 딱 맞춰 붙인다.
   -> 이름(예: '남구')으로 맞추면 여러 시도에 같은 이름이 있어서 틀릴 수 있기 때문에,
      숫자처럼 보이지만 사실은 "이름표"인 코드로 맞춘다.
4) 고령화율을 5단계로 나누어 색칠한 지도를 그린다.
5) 고령화율이 높은 시군구 10개, 낮은 시군구 10개를 표로 보여준다.
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
st.title("🗺️ 전국 시군구별 고령화 지도")
st.caption("65세 이상 인구 비율을 시군구 단위로 색칠해서 보여줍니다.")

# 데이터가 있는 인터넷 주소
POP_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/population_yearly.csv.gz"
GEO_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/boundaries/sigungu_kr.geojson"

# 색칠 구간을 나누는 경계값 (문제에서 정해준 값, 단위는 %)
BIN_EDGES = [-0.01, 19, 23, 28, 38, 100]
BIN_LABELS = ["19% 미만", "19%~23%", "23%~28%", "28%~38%", "38% 이상"]

# 낮은 값은 옅은 색, 높은 값은 진한 색이 되도록 5단계 색을 직접 정해둔다.
BIN_COLORS = {
    "19% 미만": "#fff5eb",
    "19%~23%": "#fdd0a2",
    "23%~28%": "#fd8d3c",
    "28%~38%": "#d94801",
    "38% 이상": "#7f2704",
}


# ---------------------------------------------------------
# 1. 인구 데이터 내려받고 정리하기
# ---------------------------------------------------------
@st.cache_data(show_spinner="인구 데이터를 불러오는 중입니다...")
def load_population():
    """인구 csv.gz 파일을 내려받아서 표(DataFrame)로 만든다."""
    response = requests.get(POP_URL)
    response.raise_for_status()  # 다운로드가 실패하면 여기서 바로 에러를 알려준다.

    # 압축을 풀어서 진짜 csv 내용으로 바꾼다.
    raw_bytes = gzip.decompress(response.content)

    # '코드'는 계산용 숫자가 아니라 이름표이므로, 무조건 글자(str)로 읽는다.
    # (str로 읽지 않으면 앞자리 0이 사라지는 등 문제가 생길 수 있다)
    df = pd.read_csv(io.BytesIO(raw_bytes), dtype={"코드": str})

    return df


@st.cache_data(show_spinner="시군구별 고령화율을 계산하는 중입니다...")
def make_sigungu_elderly_ratio(pop_df: pd.DataFrame):
    """
    읍면동 단위 인구 데이터를 받아서,
    - 가장 최신 연도만 남기고
    - 시군구 단위로 합쳐서
    - 65세 이상 비율(고령화율, %)을 계산한 표를 돌려준다.
    """
    # 가장 최신 연도만 사용한다.
    latest_year = pop_df["연도"].max()
    df = pop_df[pop_df["연도"] == latest_year].copy()

    # '코드'는 행정동 코드(예: 7자리 이상)이고, 앞 5자리가 시군구 코드다.
    df["시군구코드"] = df["코드"].str[:5]

    # '계_숫자세' 형태의 열 이름들을 모두 찾는다. (남녀 합산 값)
    # 예: 계_0세, 계_1세, ..., 계_99세, 계_100세 이상
    total_age_cols = [c for c in df.columns if c.startswith("계_")]

    # 그중에서 "65세 이상"에 해당하는 열만 따로 골라낸다.
    elderly_age_cols = []
    for age in range(65, 100):
        col_name = f"계_{age}세"
        if col_name in total_age_cols:
            elderly_age_cols.append(col_name)
    if "계_100세 이상" in total_age_cols:
        elderly_age_cols.append("계_100세 이상")

    # 한 읍면동의 전체 인구 = 모든 나이 열의 합
    df["전체인구"] = df[total_age_cols].sum(axis=1)
    # 한 읍면동의 65세 이상 인구 = 65세 이상 나이 열의 합
    df["고령인구"] = df[elderly_age_cols].sum(axis=1)

    # 같은 시군구에 속한 읍면동들을 하나로 합친다(전체인구, 고령인구는 더하고,
    # 시도/시군구 이름은 어차피 같은 값이 반복되니 첫 번째 값만 가져온다).
    grouped = (
        df.groupby("시군구코드")
        .agg(
            시도=("시도", "first"),
            시군구=("시군구", "first"),
            전체인구=("전체인구", "sum"),
            고령인구=("고령인구", "sum"),
        )
        .reset_index()
    )

    # 고령화율(%) = 65세 이상 인구 / 전체 인구 * 100
    grouped["고령화율"] = (grouped["고령인구"] / grouped["전체인구"] * 100).round(2)

    # 5단계 구간으로 나눈다.
    grouped["구간"] = pd.cut(
        grouped["고령화율"], bins=BIN_EDGES, labels=BIN_LABELS
    )

    return grouped, latest_year


# ---------------------------------------------------------
# 2. 지도 경계 데이터(geojson) 내려받기
# ---------------------------------------------------------
@st.cache_data(show_spinner="지도 경계 데이터를 불러오는 중입니다...")
def load_geojson():
    response = requests.get(GEO_URL)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------
# 3. 실제로 데이터를 불러와서 화면에 그리기
# ---------------------------------------------------------
population_df = load_population()
sigungu_df, latest_year = make_sigungu_elderly_ratio(population_df)
geojson_data = load_geojson()

st.subheader(f"{latest_year}년 기준 시군구별 65세 이상 인구 비율")

# ---- 지도 그리기 ----
fig = px.choropleth(
    sigungu_df,
    geojson=geojson_data,
    locations="시군구코드",          # 우리 표에서 지역을 구분하는 열
    featureidkey="properties.코드",  # geojson 쪽에서 같은 역할을 하는 속성
    color="구간",                    # 5단계 구간별로 색칠
    category_orders={"구간": BIN_LABELS},  # 범례 순서를 낮은 값 -> 높은 값 순으로 고정
    color_discrete_map=BIN_COLORS,
    hover_name="시군구",
    hover_data={
        "시도": True,
        "고령화율": ":.2f",
        "시군구코드": False,  # 지도에 뜨는 코드 값은 굳이 안 보여줘도 된다.
        "구간": False,
    },
    labels={"구간": "고령화율 구간", "고령화율": "고령화율(%)", "시도": "시도"},
)

# 배경 지도 타일 없이, 우리가 가진 경계선만 딱 맞춰서 보이게 설정한다.
fig.update_geos(fitbounds="geojson", visible=False)
fig.update_traces(marker_line_color="white", marker_line_width=0.5)
fig.update_layout(
    margin={"r": 0, "t": 10, "l": 0, "b": 0},
    legend_title_text="고령화율 구간",
    height=700,
)

st.plotly_chart(fig, use_container_width=True)

# ---- 지도 아래 순위 표 ----
st.subheader("고령화율 순위")

col_high, col_low = st.columns(2)

# 보기 좋게 표에 넣을 열만 골라서 이름도 다시 붙여준다.
display_df = sigungu_df[["시도", "시군구", "고령화율"]].copy()
display_df = display_df.rename(columns={"고령화율": "고령화율(%)"})

top10 = display_df.sort_values("고령화율(%)", ascending=False).head(10).reset_index(drop=True)
bottom10 = display_df.sort_values("고령화율(%)", ascending=True).head(10).reset_index(drop=True)

# 표에서 등수가 1부터 보이도록 인덱스를 1부터 시작하게 바꾼다.
top10.index = top10.index + 1
bottom10.index = bottom10.index + 1

with col_high:
    st.markdown("**🔺 고령화율 높은 시군구 TOP 10**")
    st.dataframe(top10, use_container_width=True)

with col_low:
    st.markdown("**🔻 고령화율 낮은 시군구 TOP 10**")
    st.dataframe(bottom10, use_container_width=True)

st.caption(
    "※ 고령화율 = (65세 이상 인구 / 전체 인구) × 100. "
    "시군구 코드(앞 5자리) 기준으로 읍·면·동 인구를 합산해서 계산했습니다."
)
