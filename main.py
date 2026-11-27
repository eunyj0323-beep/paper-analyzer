import copy
import hashlib
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import library
from analyzer import (
    API_ERRORS,
    CATEGORIES,
    DESIGNS,
    MODELS,
    SECTIONS,
    AnalysisError,
    ClaudeAnalyzer,
    GeminiAnalyzer,
    friendly_error,
)
from exporter import to_docx, to_markdown
from pdf_processor import MAX_PDF_MB, MAX_PDF_PAGES, extract_text, is_readable

load_dotenv()

st.set_page_config(page_title="논문 분석 리포트", page_icon=":material/description:", layout="wide")

MAX_FILES = 3
KST = timezone(timedelta(hours=9))


@st.cache_data(show_spinner="PDF에서 텍스트를 추출하는 중...")
def load_pdf(pdf_bytes: bytes) -> tuple[str, int]:
    return extract_text(pdf_bytes)


def make_analyzer():
    kind, model = ENGINE
    return ClaudeAnalyzer(CLAUDE_KEY, model) if kind == "claude" else GeminiAnalyzer(GEMINI_KEY)


def now() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def short(text: str, limit: int = 24) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def badge_text(text: str) -> str:
    return text.replace("[", "(").replace("]", ")")


def engine_notice() -> None:
    if ENGINE is None:
        st.warning("왼쪽 사이드바에 Anthropic API 키를 입력하면 분석을 시작할 수 있습니다.")
    elif ENGINE[0] == "gemini":
        st.caption(
            "무료 모드(Google Gemini)로 분석합니다. 논문 내용이 Google로 전송되며, 무료 등급 약관에 따라 "
            "Google 서비스 개선에 쓰일 수 있습니다. 공개 전 원고라면 왼쪽 사이드바에 본인 Claude API 키를 넣어 주세요. "
            "모든 사용자가 하루 사용량을 나눠 쓰므로 한도가 차면 다음 날 다시 쓸 수 있습니다."
        )


def usage_caption(report: dict) -> str:
    usage = report["usage"]
    tokens = f"입력 {usage['input'] + usage['cache_write'] + usage['cache_read']:,} · 출력 {usage['output']:,}"
    text = f"{report['model_label']} · {report['created'][:16].replace('T', ' ')} · 사용 토큰: {tokens}"
    if not report["free"]:
        text += f" · 대략적인 비용 ${report['cost']:.3f}"
    return text


def show_report(report: dict, key: str) -> None:
    if report["kind"] == "compare":
        st.subheader("비교 리포트")
        for n, title in enumerate(report["paper_titles"], 1):
            st.markdown(f"**논문 {n}.** {title}")
        st.caption(usage_caption(report) + " · 각 논문의 분석 리포트를 바탕으로 비교했습니다.")
        st.markdown(report["compare_text"])
    else:
        meta = report["meta"]
        st.subheader(library.display_title(report))
        byline = " · ".join(x for x in [meta["authors"], meta["year"], meta["venue"]] if x)
        if byline:
            st.caption(byline)
        badges = [f":blue-badge[{badge_text(meta['category'])}]", f":violet-badge[{meta['design']} 연구]"]
        badges += [f":gray-badge[{badge_text(k)}]" for k in meta["keywords"]]
        badges += [f":orange-badge[#{badge_text(t)}]" for t in report.get("tags", [])]
        st.markdown(" ".join(badges))
        st.caption(usage_caption(report))
        if report["chunked"]:
            st.info("분량이 많아 본문을 나눠 압축한 뒤 분석했습니다.")
        if report.get("pdf_mode"):
            st.info("PDF에서 글자를 뽑을 수 없어 AI가 PDF 파일을 직접 읽고 분석했습니다.")
        tabs = st.tabs([title for _, title, _ in SECTIONS])
        for tab, (section_key, _, _) in zip(tabs, SECTIONS):
            with tab:
                section = report["sections"][section_key]
                if section["error"]:
                    st.error(section["error"])
                else:
                    st.markdown(section["text"])

    markdown = to_markdown(report)
    stem = "비교_리포트" if report["kind"] == "compare" else Path(report["filename"]).stem + "_분석"
    col1, col2 = st.columns(2)
    col1.download_button(
        "마크다운(.md) 다운로드", markdown, file_name=f"{stem}.md", mime="text/markdown",
        on_click="ignore", width="stretch", key=f"{key}_md", icon=":material/download:",
    )
    col2.download_button(
        "워드(.docx) 다운로드", to_docx(markdown), file_name=f"{stem}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        on_click="ignore", width="stretch", key=f"{key}_docx", icon=":material/download:",
    )


def paper_report(paper: dict, analyzer, sections: dict, meta: dict) -> dict:
    if not meta["title"]:
        meta["title"] = Path(paper["name"]).stem
    return {
        "id": paper["id"],
        "kind": "paper",
        "filename": paper["name"],
        "created": now(),
        "model_label": analyzer.label,
        "free": isinstance(analyzer, GeminiAnalyzer),
        "chunked": analyzer.chunked,
        "pdf_mode": paper["pdf_mode"],
        "usage": analyzer.usage,
        "cost": analyzer.estimated_cost(),
        "meta": meta,
        "tags": [],
        "sections": sections,
    }


def compare_id(reports: list[dict]) -> str:
    return "c-" + hashlib.sha256("|".join(sorted(r["id"] for r in reports)).encode()).hexdigest()[:16]


def run_compare(reports: list[dict], progress_callback) -> dict:
    """Raises API_ERRORS or AnalysisError; the caller shows them."""
    analyzer = make_analyzer()
    text = analyzer.compare(reports, progress_callback)
    titles = [library.display_title(r) for r in reports]
    report = {
        "id": compare_id(reports),
        "kind": "compare",
        "filename": "비교 리포트",
        "created": now(),
        "model_label": analyzer.label,
        "free": isinstance(analyzer, GeminiAnalyzer),
        "chunked": False,
        "usage": analyzer.usage,
        "cost": analyzer.estimated_cost(),
        "meta": {"title": "비교: " + " / ".join(short(t, 30) for t in titles), "authors": "", "year": "",
                 "venue": "", "category": library.COMPARE_CATEGORY, "design": "", "keywords": []},
        "tags": [],
        "paper_ids": [r["id"] for r in reports],
        "paper_titles": titles,
        "compare_text": text,
    }
    library.save(report)
    return report


def start_new_analysis() -> None:
    st.session_state.pop("current", None)
    st.session_state["uploader_n"] = st.session_state.get("uploader_n", 0) + 1


# ---------------------------------------------------------------- 분석하기

def read_uploads(uploaded) -> tuple[list[dict], list[str]]:
    papers, problems, seen = [], [], set()
    for file in uploaded:
        pdf_bytes = file.getvalue()
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        try:
            text, pages = load_pdf(pdf_bytes)
        except ValueError as e:
            problems.append(f"**{file.name}**: {e}")
            continue
        pdf_mode = not is_readable(text)
        if pdf_mode and (pages > MAX_PDF_PAGES or len(pdf_bytes) > MAX_PDF_MB * 1024 * 1024):
            problems.append(
                f"**{file.name}**: 글자를 뽑을 수 없는 PDF(스캔본이거나 글꼴이 깨진 파일)입니다. 이런 파일은 AI가 PDF를 직접 "
                f"읽어야 하는데, {MAX_PDF_PAGES}쪽·{MAX_PDF_MB}MB 이하만 가능합니다. 필요한 쪽만 잘라서 올려 주세요."
            )
            continue
        papers.append({
            "id": "p-" + digest[:16], "name": file.name, "bytes": pdf_bytes, "text": text,
            "pages": pages, "pdf_mode": pdf_mode, "garbled": not pdf_mode and text.count("(cid:") > 20,
        })
    return papers, problems


def analyze_page() -> None:
    current = st.session_state.get("current")
    head, button = st.columns([5, 1], vertical_alignment="bottom")
    head.title("논문 분석 리포트")
    button.button("새 분석 시작", icon=":material/add:", width="stretch", on_click=start_new_analysis)
    st.caption(
        "논문 PDF를 올리면 요약, 연구방법·통계기법, 인용·참고문헌, 차별점·갭 분석을 한국어 리포트로 정리합니다. "
        f"한국어·영어 논문 모두 됩니다. 2~{MAX_FILES}편을 함께 올리면 논문끼리 비교한 리포트도 만들어 드려요. "
        "분석한 리포트는 **내 보관함**에 자동으로 저장됩니다."
    )

    uploaded = st.file_uploader(
        f"논문 PDF (최대 {MAX_FILES}편)", type="pdf", accept_multiple_files=True,
        key=f"uploader_{st.session_state.get('uploader_n', 0)}",
    )
    if not uploaded:
        st.info("분석할 논문 PDF를 올려 주세요.")
        return
    if len(uploaded) > MAX_FILES:
        st.error(f"한 번에 {MAX_FILES}편까지 분석할 수 있습니다. 파일 옆 X를 눌러 {len(uploaded) - MAX_FILES}편을 빼 주세요.")
        return

    papers, problems = read_uploads(uploaded)
    for problem in problems:
        st.error(problem)
    if not papers:
        return

    st.dataframe(
        pd.DataFrame([{
            "파일": p["name"],
            "페이지": p["pages"],
            "추출된 글자 수": len(p["text"]),
            "읽는 방식": "PDF 직접 읽기 (스캔본·글자 깨짐)" if p["pdf_mode"] else "텍스트 추출",
        } for p in papers]),
        hide_index=True, width="stretch",
    )
    if any(p["pdf_mode"] for p in papers):
        st.info(
            "글자를 뽑을 수 없는 PDF(스캔본이거나 글꼴이 깨진 파일)가 있어서, 그 논문은 AI가 PDF 파일을 직접 읽고 분석합니다. "
            "시간이 조금 더 걸릴 수 있습니다."
        )
    for p in papers:
        if p["garbled"]:
            st.warning(f"{p['name']}: 일부 글자가 깨진 채로 추출되었습니다. 분석 결과에 빠진 내용이 있을 수 있습니다.")

    batch_key = (tuple(p["id"] for p in papers), ENGINE)
    if current and current["key"] != batch_key:
        current = None

    saved = st.session_state.get("library") or {}
    already = [p for p in papers if p["id"] in saved]
    reuse = False
    if already and not current:  # after an analysis the button means "analyze again", so don't reuse
        reuse = st.checkbox(
            f"보관함에 이미 있는 논문({len(already)}편)은 저장된 분석 결과를 그대로 쓰기",
            value=True, help="다시 분석하지 않으므로 시간과 사용량(비용)을 아낍니다.",
        )

    engine_notice()
    if len(papers) == 1:
        label = "다시 분석" if current else "분석 시작"
    else:
        label = f"{len(papers)}편 {'다시 ' if current else ''}분석하고 비교 리포트 만들기"
    if st.button(label, type="primary", disabled=ENGINE is None, icon=":material/play_arrow:"):
        st.session_state["current"] = analyze_batch(papers, batch_key, saved if reuse else {})
        st.rerun()

    if current:
        show_current(current)


def analyze_batch(papers: list[dict], batch_key, reusable: dict) -> dict:
    steps = len(papers) + (1 if len(papers) > 1 else 0)
    progress = st.progress(0.0, text="분석 준비 중...")
    reports, errors = [], []
    stopped = False
    for i, paper in enumerate(papers):
        if paper["id"] in reusable:
            reports.append(reusable[paper["id"]])
            continue
        prefix = f"[{i + 1}/{len(papers)}] {short(paper['name'], 30)} · " if len(papers) > 1 else ""

        def on_progress(message, fraction, i=i, prefix=prefix):
            progress.progress(min((i + fraction) / steps, 1.0), text=prefix + message)

        analyzer = make_analyzer()
        try:
            sections, meta = analyzer.analyze(
                paper["text"], on_progress, pdf_bytes=paper["bytes"] if paper["pdf_mode"] else None
            )
        except API_ERRORS as e:
            errors.append(f"**{paper['name']}**: {friendly_error(e)}")
            stopped = True  # bad key, quota, outage: the next papers would fail the same way
            break
        except AnalysisError as e:
            errors.append(f"**{paper['name']}**: {e}")
            continue
        report = paper_report(paper, analyzer, sections, meta)
        library.save(report)
        reports.append(report)

    compare = None
    if len(papers) > 1 and stopped:
        errors.append("분석하지 못한 논문이 있어 비교 리포트를 만들지 않았습니다.")
    elif len(papers) > 1 and len(reports) < 2:
        errors.append("비교하려면 분석된 논문이 2편 이상 필요합니다.")
    elif len(papers) > 1:
        existing = reusable.get(compare_id(reports))
        if existing and all(r["id"] in reusable for r in reports):
            compare = existing
        else:
            def on_progress(message, fraction):
                progress.progress(min((steps - 1 + fraction) / steps, 1.0), text=message)
            try:
                compare = run_compare(reports, on_progress)
            except API_ERRORS as e:
                errors.append(f"**비교 리포트**: {friendly_error(e)}")
            except AnalysisError as e:
                errors.append(f"**비교 리포트**: {e}")
    progress.empty()
    return {"key": batch_key, "reports": reports, "compare": compare, "errors": errors}


def show_current(current: dict) -> None:
    for error in current["errors"]:
        st.error(error)
    views = ([("비교 리포트", current["compare"])] if current["compare"] else []) + [
        (f"논문 {n}: {short(library.display_title(r))}", r) for n, r in enumerate(current["reports"], 1)
    ]
    if not views:
        return
    st.divider()
    st.success("분석이 끝났습니다. 결과는 내 보관함에 자동으로 저장되었습니다.", icon=":material/check_circle:")
    choice = 0
    if len(views) > 1:
        choice = st.segmented_control(
            "보기", range(len(views)), format_func=lambda i: views[i][0], default=0,
            key=f"view_{hash(current['key'])}",
        ) or 0
    show_report(views[choice][1], key=f"current_{choice}")


# ---------------------------------------------------------------- 내 보관함

def library_page() -> None:
    st.title("내 보관함")
    st.caption(
        "분석한 리포트가 자동으로 저장됩니다. 보관함은 **지금 쓰는 브라우저에만** 저장되므로 다른 사람은 볼 수 없고, "
        "다른 기기·브라우저에서는 보이지 않습니다. 브라우저의 쿠키·사이트 데이터를 지우면 사라지니 가끔 아래에서 백업 파일을 내려받아 두세요."
    )
    reports = st.session_state.get("library")
    if reports is None:
        st.info("보관함을 불러오는 중입니다...")
        return
    if st.session_state.get("library_error"):
        st.warning(st.session_state["library_error"])
    if not reports:
        st.info("아직 저장된 리포트가 없습니다. 분석하기 화면에서 논문을 분석하면 여기에 쌓입니다.")
        st.page_link(ANALYZE_PAGE, label="논문 분석하러 가기", icon=":material/upload_file:")
        backup_section(reports)
        return

    query_col, scope_col = st.columns([3, 1])
    query = query_col.text_input("검색", placeholder="예: 팬 충성도, 구조방정식, PROCESS, 2023, 저자 이름")
    scope = scope_col.selectbox("검색 범위", list(library.SEARCH_SCOPES))

    counts = Counter(library.category_of(r) for r in reports.values())
    categories = [c for c in CATEGORIES + [library.COMPARE_CATEGORY] if counts[c]]
    picked_categories = st.pills(
        "분야", categories, selection_mode="multi", format_func=lambda c: f"{c} ({counts[c]})"
    )
    design_counts = Counter(r["meta"]["design"] for r in reports.values() if r["kind"] == "paper")
    picked_designs = st.pills(
        "연구유형", [d for d in DESIGNS if design_counts[d]], selection_mode="multi",
        format_func=lambda d: f"{d} ({design_counts[d]})",
    )
    tag_counts = Counter(t for r in reports.values() for t in r.get("tags", []))
    picked_tags = []
    if tag_counts:
        picked_tags = st.pills(
            "내 태그", sorted(tag_counts), selection_mode="multi", format_func=lambda t: f"#{t} ({tag_counts[t]})"
        )

    found = []
    for report in sorted(reports.values(), key=lambda r: r["created"], reverse=True):
        if picked_categories and library.category_of(report) not in picked_categories:
            continue
        if picked_designs and (report["kind"] != "paper" or report["meta"]["design"] not in picked_designs):
            continue
        if picked_tags and not set(picked_tags) & set(report.get("tags", [])):
            continue
        snippet = library.matches(report, query, scope)
        if snippet is not None:
            found.append((report, snippet))

    st.caption(f"전체 {len(reports)}개 중 {len(found)}개 · 표에서 행 왼쪽을 눌러 선택하세요. "
               "1개를 고르면 리포트를 보고, 논문 2~3개를 고르면 비교 리포트를 만들 수 있습니다.")
    rows = [{
        "제목": library.display_title(r),
        "저자·연도": " ".join(x for x in [short(r["meta"]["authors"], 20), f"({r['meta']['year']})" if r["meta"]["year"] else ""] if x),
        "분야": library.category_of(r),
        "연구유형": r["meta"]["design"],
        "키워드": ", ".join(r["meta"]["keywords"]),
        "내 태그": ", ".join(r.get("tags", [])),
        "분석일": r["created"][:10],
        **({"검색어가 나온 부분": snippet} if query.strip() else {}),
    } for r, snippet in found]
    table_key = "lib_table_" + hashlib.md5(repr((query, scope, picked_categories, picked_designs, picked_tags)).encode()).hexdigest()
    event = st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch", on_select="rerun",
        selection_mode="multi-row", key=table_key,
        column_config={"제목": st.column_config.TextColumn(width="large")},
    )
    selected = [found[i][0] for i in event.selection.rows if i < len(found)]

    if len(selected) == 1:
        st.divider()
        show_report(selected[0], key=f"lib_{selected[0]['id']}")
        edit_section(selected[0])
    elif len(selected) > 1:
        st.divider()
        compare_section(selected)

    st.divider()
    backup_section(reports)


def edit_section(report: dict) -> None:
    with st.expander("분류 고치기 · 삭제", icon=":material/edit:"):
        with st.form(f"edit_{report['id']}"):
            title = st.text_input("제목", value=library.display_title(report))
            if report["kind"] == "paper":
                left, right = st.columns(2)
                category = left.selectbox("분야", CATEGORIES, index=CATEGORIES.index(report["meta"]["category"]))
                design = right.selectbox("연구유형", DESIGNS, index=DESIGNS.index(report["meta"]["design"]))
                keywords = st.text_input("키워드 (쉼표로 구분)", value=", ".join(report["meta"]["keywords"]))
            tags = st.text_input(
                "내 태그 (쉼표로 구분)", value=", ".join(report.get("tags", [])),
                placeholder="예: 졸업논문, 2장 선행연구, 수업 발표",
            )
            if st.form_submit_button("저장", type="primary"):
                updated = copy.deepcopy(report)
                updated["meta"]["title"] = title.strip() or report["meta"]["title"]
                if report["kind"] == "paper":
                    updated["meta"].update(
                        category=category, design=design,
                        keywords=[k.strip() for k in keywords.split(",") if k.strip()],
                    )
                updated["tags"] = list(dict.fromkeys(t.strip().lstrip("#") for t in tags.split(",") if t.strip()))
                library.save(updated)
                st.toast("저장했습니다.", icon=":material/check:")
                st.rerun()
        with st.popover("이 리포트 삭제", icon=":material/delete:"):
            st.write("보관함에서 이 리포트를 지웁니다. 되돌릴 수 없습니다.")
            if st.button("삭제", type="primary", key=f"delete_{report['id']}"):
                library.delete(report["id"])
                st.toast("삭제했습니다.", icon=":material/delete:")
                st.rerun()


def compare_section(selected: list[dict]) -> None:
    papers = [r for r in selected if r["kind"] == "paper"]
    if len(papers) != len(selected) or len(papers) > MAX_FILES:
        st.warning(f"비교 리포트는 논문 리포트 2~{MAX_FILES}개를 선택했을 때 만들 수 있습니다. (비교 리포트끼리는 비교할 수 없습니다)")
        return
    st.subheader(f"선택한 논문 {len(papers)}편 비교")
    for n, report in enumerate(papers, 1):
        st.markdown(f"**논문 {n}.** {library.display_title(report)}")
    existing = (st.session_state.get("library") or {}).get(compare_id(papers))
    engine_notice()
    label = "다시 비교하기" if existing else "비교 리포트 만들기"
    if st.button(label, type="primary", disabled=ENGINE is None, icon=":material/compare_arrows:"):
        progress = st.progress(0.0, text="비교 준비 중...")
        try:
            existing = run_compare(papers, lambda message, fraction: progress.progress(fraction, text=message))
        except API_ERRORS as e:
            st.error(friendly_error(e))
        except AnalysisError as e:
            st.error(str(e))
        progress.empty()
    if existing:
        show_report(existing, key=f"lib_{existing['id']}")


def backup_section(reports: dict) -> None:
    with st.expander("보관함 백업 · 복원", icon=":material/backup:"):
        st.write("다른 컴퓨터나 브라우저로 옮기거나 혹시 모를 삭제에 대비해 보관함 전체를 파일 하나로 내려받을 수 있습니다.")
        st.download_button(
            "보관함 백업 파일 내려받기 (.json)", library.export_json(reports),
            file_name=f"논문보관함_백업_{datetime.now(KST):%Y%m%d}.json", mime="application/json",
            on_click="ignore", disabled=not reports, icon=":material/download:",
        )
        restore_n = st.session_state.get("restore_n", 0)
        backup = st.file_uploader("백업 파일로 복원 (지금 보관함에 합쳐집니다)", type="json", key=f"restore_{restore_n}")
        if backup and st.button("복원하기", icon=":material/upload:"):
            try:
                count = library.import_json(backup.getvalue())
            except (ValueError, UnicodeDecodeError) as e:
                st.error(f"복원하지 못했습니다: {e}")
            else:
                st.session_state["restore_n"] = restore_n + 1
                st.toast(f"리포트 {count}개를 복원했습니다.", icon=":material/check:")
                st.rerun()


# ---------------------------------------------------------------- 공통

with st.sidebar:
    st.header("설정")
    CLAUDE_KEY = st.text_input(
        "Anthropic API 키 (선택)",
        type="password",
        placeholder="sk-ant-...",
        help="입력하면 Claude로 분석합니다. 비워 두면 무료 모드(Google Gemini)로 분석합니다.",
    ) or os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
    if CLAUDE_KEY:
        model = st.selectbox("모델", list(MODELS), format_func=lambda m: MODELS[m]["label"])
        st.caption(
            "키는 저장되지 않고 지금 열린 브라우저 탭에서만 쓰입니다. "
            "분석 비용은 이 키의 계정에 청구됩니다."
        )
        ENGINE = ("claude", model)
    elif GEMINI_KEY:
        st.caption("키를 비워 두면 무료 모드(Google Gemini)로 분석합니다.")
        ENGINE = ("gemini", None)
    else:
        ENGINE = None
    st.markdown("[Claude API 키 발급받기](https://platform.claude.com/settings/keys)")
    library.sync()

ANALYZE_PAGE = st.Page(analyze_page, title="분석하기", icon=":material/upload_file:", url_path="analyze", default=True)
LIBRARY_PAGE = st.Page(library_page, title="내 보관함", icon=":material/folder_open:", url_path="library")
st.navigation([ANALYZE_PAGE, LIBRARY_PAGE], position="top").run()
