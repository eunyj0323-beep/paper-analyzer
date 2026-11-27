import io
import re

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt

from analyzer import SECTIONS

FONT = "맑은 고딕"
FOOTER = "*AI가 자동으로 생성한 분석입니다. 인용 전 원문과 대조해 확인하세요.*"


def to_markdown(report: dict) -> str:
    if report["kind"] == "compare":
        return _compare_markdown(report)
    meta = report["meta"]
    lines = [f"# 논문 분석 리포트: {meta['title'] or report['filename']}", ""]
    if meta["authors"] or meta["year"]:
        lines.append(f"- 저자·연도: {meta['authors']} ({meta['year'] or '연도 미상'})")
    lines += [
        f"- 파일: {report['filename']}",
        f"- 분야: {meta['category']} · 연구유형: {meta['design']}",
    ]
    if meta["keywords"]:
        lines.append(f"- 키워드: {', '.join(meta['keywords'])}")
    if report.get("tags"):
        lines.append(f"- 내 태그: {', '.join(report['tags'])}")
    lines += [
        f"- 분석 모델: {report['model_label']}",
        f"- 분석 날짜: {report['created'][:10]}",
    ]
    if report["chunked"]:
        lines.append("- 분량이 많아 본문을 나눠 압축한 뒤 분석했습니다.")
    if report.get("pdf_mode"):
        lines.append("- PDF에서 글자를 뽑을 수 없어 AI가 PDF 파일을 직접 읽고 분석했습니다.")
    for n, (key, title, _) in enumerate(SECTIONS, 1):
        section = report["sections"][key]
        body = section["text"] if section["text"] else f"(분석 실패: {section['error']})"
        lines += ["", "---", "", f"## {n}. {title}", "", body]
    lines += ["", "---", "", FOOTER, ""]
    return "\n".join(lines)


def _compare_markdown(report: dict) -> str:
    lines = ["# 논문 비교 리포트", ""]
    for n, title in enumerate(report["paper_titles"], 1):
        lines.append(f"- 논문 {n}: {title}")
    lines += [
        f"- 분석 모델: {report['model_label']}",
        f"- 분석 날짜: {report['created'][:10]}",
        "- 각 논문의 분석 리포트를 바탕으로 비교했습니다.",
        "", "---", "",
        report["compare_text"],
        "", "---", "", FOOTER, "",
    ]
    return "\n".join(lines)


def to_docx(markdown: str) -> bytes:
    doc = Document()
    for style_name in ["Normal", "Title", "Heading 1", "Heading 2", "Heading 3"]:
        style = doc.styles[style_name]
        style.font.name = FONT
        style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    doc.styles["Normal"].font.size = Pt(10.5)

    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            _add_table(doc, rows)
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)", stripped)
        if heading:
            level = len(heading.group(1)) - 1
            doc.add_heading(heading.group(2).replace("**", ""), level=level)
        elif re.match(r"^[-*]\s+", stripped) and stripped != "---":
            indent = (len(line) - len(line.lstrip())) // 2
            style = "List Bullet 2" if indent else "List Bullet"
            _add_inline(doc.add_paragraph(style=style), re.sub(r"^[-*]\s+", "", stripped))
        elif stripped.startswith(">"):
            paragraph = doc.add_paragraph()
            paragraph.add_run(stripped.lstrip("> ")).italic = True
        elif stripped and stripped != "---":
            _add_inline(doc.add_paragraph(), stripped)
        i += 1

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _add_inline(paragraph, text: str) -> None:
    for part in re.split(r"(\*\*[^*]+\*\*)", text):
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            paragraph.add_run(part[2:-2]).bold = True
        elif part:
            paragraph.add_run(part.replace("`", ""))


def _add_table(doc, rows: list[str]) -> None:
    cells = [
        [c.strip() for c in row.strip("|").split("|")]
        for row in rows
        if not re.fullmatch(r"\|?[\s:\-|]+\|?", row)
    ]
    if not cells:
        return
    width = max(len(r) for r in cells)
    table = doc.add_table(rows=len(cells), cols=width)
    table.style = "Table Grid"
    for r, row in enumerate(cells):
        for c, value in enumerate(row):
            paragraph = table.cell(r, c).paragraphs[0]
            _add_inline(paragraph, value)
            if r == 0:
                for run in paragraph.runs:
                    run.bold = True
    doc.add_paragraph()
