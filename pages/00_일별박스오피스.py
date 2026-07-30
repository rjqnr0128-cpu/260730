import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

st.set_page_config(page_title="박스오피스 대시보드", layout="wide")
st.title("🎬 일별 박스오피스 대시보드")

# 비밀 금고에서 인증키 꺼내기
KOBIS_KEY = st.secrets["KOBIS_KEY"]

# 달력에서 날짜 선택 (가장 늦은 날짜는 어제까지)
yesterday = datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(days=1)
selected_date = st.date_input(
    "조회할 날짜를 선택하세요",
    value=yesterday,
    max_value=yesterday
)

target_dt = selected_date.strftime("%Y%m%d")
st.caption(f"조회 기준일: {selected_date.strftime('%Y-%m-%d')}")

url = "https://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"

@st.cache_data(ttl=3600)
def get_box_office(dt, key):
    try:
        res = requests.get(url, params={"key": key, "targetDt": dt}, timeout=10)
        if res.status_code != 200:
            return None
        return res.json()
    except:
        return None

data = get_box_office(target_dt, KOBIS_KEY)

if not data or "faultInfo" in data:
    st.error("인증키가 올바르지 않거나 API 요청에 실패했습니다.")
    st.stop()

box_list = data.get("boxOfficeResult", {}).get("dailyBoxOfficeList", [])

# 고른 날짜에 영화 목록이 비어 있으면 안내
if not box_list:
    st.warning("그날은 아직 집계 전입니다.")
    st.stop()

df = pd.DataFrame(box_list)

# 숫자형 변환
for col in ["rank", "audiCnt", "audiAcc", "scrnCnt", "showCnt", "rankInten"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# 1위 영화 지표 카드 세 장
top = df.sort_values("rank").iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("선택일 1위", top["movieNm"])
c2.metric("관객수", f"{int(top['audiCnt']):,}명")
c3.metric("누적 관객", f"{int(top['audiAcc']):,}명")

# 1위 영화 포스터 크게 보여주기 영역
st.subheader("🖼️ 1위 영화 포스터")
top_movie_name = top["movieNm"]
st.write(f"**{top_movie_name}** (선택일 가장 많은 사람이 본 영화)")
st.info("💡 TMDB API 등 외부 이미지 소스를 연동하여 포스터 이미지를 함께 출력하실 수 있습니다.")

# 조건에 따른 영화명 이모지 및 순위 증감 화살표 가공
def decorate_movie_name(row):
    name = row["movieNm"]
    audi_acc = row["audiAcc"]
    if audi_acc >= 1000000:
        name += " 🏆"
    elif audi_acc < 10000:
        name += " 💀"
    return name

def format_rank_inten(val):
    v = int(val)
    if v > 0:
        return f"🔴 ▲ {v}"
    elif v < 0:
        return f"🔵 ▼ {abs(v)}"
    else:
        return "-"

df["영화명_표시"] = df.apply(decorate_movie_name, axis=1)
df["순위증감_표시"] = df["rankInten"].apply(format_rank_inten)

# 표 정리
table = df[["rank", "순위증감_표시", "영화명_표시", "openDt", "audiCnt", "audiAcc", "scrnCnt"]].copy()
table.columns = ["순위", "전일대비", "영화명", "개봉일", "관객수", "누적관객", "스크린수"]
table = table.sort_values("순위").reset_index(drop=True)

st.subheader("📋 박스오피스 TOP 10")
st.dataframe(table, use_container_width=True)

st.subheader("📈 관객수 상위 5편")
top5 = df.sort_values("audiCnt", ascending=False).head(5)
st.bar_chart(top5.set_index("movieNm")["audiCnt"])
