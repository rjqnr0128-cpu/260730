import streamlit as st
from openai import OpenAI

# 페이지 기본 설정 (sidebar를 활용하기 위해 layout은 wide 또는 기본값 사용)
st.set_page_config(page_title="AI 정보 선생님", page_icon="🤖", layout="wide")

st.title("🤖 AI 정보 선생님")

# ----------------------------------------------------
# 📌 오른쪽(사이드바) 댓글/방명록 공간 만들기
# ----------------------------------------------------
with st.sidebar:
    st.header("💬 방문객 방명록")
    st.write("선생님께 한 마디씩 남겨주세요!")

    # 세션 상태에 댓글 목록이 없으면 초기화
    if "comments" not in st.session_state:
        st.session_state.comments = []

    # 댓글 입력 폼
    with st.form("comment_form", clear_on_submit=True):
        writer = st.text_input("이름 (닉네임)", placeholder="홍길동")
        content = st.text_input("댓글 내용", placeholder="좋은 가르침 감사합니다!")
        submitted = st.form_submit_button("댓글 남기기")

        if submitted:
            if writer.strip() and content.strip():
                st.session_state.comments.append({"writer": writer, "content": content})
                st.success("댓글이 등록되었습니다!")
            else:
                st.warning("이름과 내용을 모두 입력해 주세요.")

    st.divider()

    # 남겨진 댓글들을 아래로 스크롤하며 보여주기
    st.subheader("📝 최근 댓글")
    if st.session_state.comments:
        # 최신 글이 위로 오도록 역순 출력
        for c in reversed(st.session_state.comments):
            st.markdown(f"**{c['writer']}**\n> {c['content']}")
    else:
        st.info("아직 등록된 댓글이 없습니다. 첫 댓글을 남겨보세요!")


# ----------------------------------------------------
# 🤖 AI 챗봇 본문 영역
# ----------------------------------------------------

# 비밀 금고(secrets)에서 API 키를 꺼내 접속 준비
client = OpenAI(
    api_key=st.secrets["SOLAR_API_KEY"],
    base_url="https://api.upstage.ai/v1",
)

# AI의 성격 (화면에는 띄우지 않고 요청에만 함께 보낸다)
SYSTEM_PROMPT = (
    "너는 중고등학생에게 설명하는 친절한 정보 선생님이야. "
    "어려운 말은 쉬운 말로 바꿔 주고, 반드시 순수 한국어로만 답해"
)

# 대화 기록이 없으면 처음 한 번만 만들어 둔다
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# 지금까지의 대화를 말풍선으로 다시 그리기 (성격 문장은 숨김)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 채팅 입력창
user_input = st.chat_input("궁금한 것을 물어보세요!")

if user_input:
    # 보낸 말을 기록에 넣고 화면에도 그리기
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI 답 받아오기 (실패하면 빨간 오류 화면 대신 안내 문구)
    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model="solar-open2",                           # 모델 이름은 그대로 유지
                messages=st.session_state.messages,  # 대화 전체를 함께 보내 기억 유지
                reasoning_effort="none",             # 추론 끄기 -> 바로 답변 시작
                stream=True,                         # 글자가 실시간으로 흐르게
            )
            answer = st.write_stream(
                chunk.choices[0].delta.content or ""
                for chunk in stream if chunk.choices
            )
            # AI 답도 기록에 저장 (다음 질문에 이어서 사용)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception:
            st.error("응답을 받지 못했습니다. 잠시 후 다시 보내 주세요.")
