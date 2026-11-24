import hashlib
import os
from pathlib import Path

import anthropic
import streamlit as st
from dotenv import load_dotenv

from analyzer import MODELS, SECTIONS, PaperAnalyzer, friendly_error
from exporter import to_docx, to_markdown
from pdf_processor import extract_text

load_dotenv()

st.set_page_config(page_title="논문 분석 리포트", page_icon=":material/description:", layout="wide")


@st.cache_data(show_spinner="PDF에서 텍스트를 추출하는 중...")
def load_pdf(pdf_bytes: bytes) -> tuple[str, int]:
    return extract_text(pdf_bytes)


with st.sidebar:
    st.header("설정")
    api_key = st.text_input(
        "Anthropic API 키",
        type="password",
        placeholder="sk-ant-...",
        help="Claude Console에서 발급받은 키를 붙여넣으세요.",
    ) or os.getenv("ANTHROPIC_API_KEY", "")
    st.caption(
        "키는 저장되지 않고 지금 열린 브라우저 탭에서만 쓰입니다. "
        "분석 비용은 이 키의 계정에 청구됩니다."
    )
    st.markdown("[API 키 발급받기](https://platform.claude.com/settings/keys)")
    model = st.selectbox("모델", list(MODELS), format_func=lambda m: MODELS[m]["label"])

st.title("논문 분석 리포트")
st.caption(
    "논문 PDF를 올리면 요약, 연구방법·통계기법, 인용·참고문헌, 차별점·갭 분석을 "
    "한국어 리포트로 정리합니다. 한국어·영어 논문 모두 됩니다."
)

uploaded = st.file_uploader("논문 PDF", type="pdf")
if uploaded is None:
    st.info("분석할 논문 PDF를 올려 주세요.")
    st.stop()

pdf_bytes = uploaded.getvalue()
try:
    text, page_count = load_pdf(pdf_bytes)
except Exception as e:
    st.error(f"PDF를 읽지 못했습니다: {e}")
    st.stop()

if len(text) < 200:
    st.error("PDF에서 글자를 거의 추출하지 못했습니다. 스캔한 이미지 PDF는 지원하지 않습니다.")
    st.stop()

col1, col2 = st.columns(2)
col1.metric("페이지", page_count)
col2.metric("추출된 글자 수", f"{len(text):,}")
if text.count("(cid:") > 20:
    st.warning("일부 글자가 깨진 채로 추출되었습니다. 분석 결과에 빠진 내용이 있을 수 있습니다.")

result_key = (hashlib.sha256(pdf_bytes).hexdigest(), model)
result = st.session_state.get("result")
if result and result["key"] != result_key:
    result = None

if not api_key:
    st.warning("왼쪽 사이드바에 API 키를 입력하면 분석을 시작할 수 있습니다.")

if st.button("다시 분석" if result else "분석 시작", type="primary", disabled=not api_key):
    analyzer = PaperAnalyzer(api_key, model)
    progress = st.progress(0.0, text="분석 준비 중...")
    try:
        sections = analyzer.analyze(text, lambda message, fraction: progress.progress(fraction, text=message))
    except anthropic.APIError as e:
        progress.empty()
        st.error(friendly_error(e))
        st.stop()
    progress.empty()
    result = {
        "key": result_key,
        "filename": uploaded.name,
        "model": model,
        "sections": sections,
        "chunked": analyzer.chunked,
        "usage": analyzer.usage,
        "cost": analyzer.estimated_cost(),
    }
    st.session_state["result"] = result
    st.rerun()

if not result:
    st.stop()

usage = result["usage"]
st.caption(
    f"사용 토큰: 입력 {usage['input'] + usage['cache_write'] + usage['cache_read']:,} · "
    f"출력 {usage['output']:,} · 대략적인 비용 ${result['cost']:.3f}"
)
if result["chunked"]:
    st.info("분량이 많아 본문을 나눠 압축한 뒤 분석했습니다.")

tabs = st.tabs([title for _, title, _ in SECTIONS])
for tab, (key, _, _) in zip(tabs, SECTIONS):
    with tab:
        section = result["sections"][key]
        if section["error"]:
            st.error(section["error"])
        else:
            st.markdown(section["text"])

markdown = to_markdown(result)
stem = Path(result["filename"]).stem
col1, col2 = st.columns(2)
col1.download_button(
    "마크다운(.md) 다운로드",
    markdown,
    file_name=f"{stem}_분석.md",
    mime="text/markdown",
    on_click="ignore",
    width="stretch",
)
col2.download_button(
    "워드(.docx) 다운로드",
    to_docx(markdown),
    file_name=f"{stem}_분석.docx",
    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    on_click="ignore",
    width="stretch",
)
