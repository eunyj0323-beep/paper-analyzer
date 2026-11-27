"""Saved reports, kept in each visitor's own browser (localStorage).

The hosted app has no per-user storage, and saving everyone's reports on the server would
mix classmates' papers together, so each browser keeps its own library. Python sends
changes as ops; the browser applies them and sends back the whole library.
"""

import json
import re
import uuid

import streamlit as st

from analyzer import SECTIONS

STORAGE_KEY = "paper-analyzer-library-v1"
COMPARE_CATEGORY = "비교 리포트"

_JS = """
export default function (component) {
  const { data, setStateValue } = component;
  const KEY = "%s";
  const acked = (window.__paperLibraryAcked ||= new Set());
  const ops = ((data && data.ops) || []).filter((op) => !acked.has(op.op_id));
  if (window.__paperLibraryLoaded && ops.length === 0) return;

  let reports = {};
  let error = null;
  try {
    reports = JSON.parse(localStorage.getItem(KEY) || "{}");
  } catch (e) {
    error = "보관함을 읽지 못했습니다: " + e;
  }
  if (ops.length) {
    for (const op of ops) {
      if (op.type === "put") reports[op.report.id] = op.report;
      else if (op.type === "delete") delete reports[op.id];
    }
    try {
      localStorage.setItem(KEY, JSON.stringify(reports));
    } catch (e) {
      error = "브라우저 저장 공간이 부족하거나 저장이 막혀 있어 보관함에 저장하지 못했습니다. (" + e.name + ")";
    }
    ops.forEach((op) => acked.add(op.op_id));
  }
  window.__paperLibraryLoaded = true;
  setStateValue("snapshot", { reports, acked: ops.map((op) => op.op_id), error });
}
""" % STORAGE_KEY

_storage = st.components.v2.component("paper_library_storage", js=_JS)


def sync() -> dict | None:
    """Mount the storage component (call once per run). Returns {id: report}, or None while loading."""
    pending = st.session_state.setdefault("library_ops", [])
    result = _storage(
        key="paper_library_storage",
        data={"ops": pending},
        default={"snapshot": None},
        on_snapshot_change=lambda: None,
        height=0,
    )
    snapshot = result.snapshot
    if snapshot is None:
        return None
    acked = set(snapshot.get("acked") or [])
    pending[:] = [op for op in pending if op["op_id"] not in acked]
    st.session_state["library_error"] = snapshot.get("error")

    reports = dict(snapshot.get("reports") or {})
    for op in pending:  # not yet confirmed by the browser, but show them already
        if op["type"] == "put":
            reports[op["report"]["id"]] = op["report"]
        else:
            reports.pop(op["id"], None)
    st.session_state["library"] = reports
    return reports


def save(report: dict) -> None:
    _queue({"type": "put", "report": report})
    st.session_state.setdefault("library", {})[report["id"]] = report


def delete(report_id: str) -> None:
    _queue({"type": "delete", "id": report_id})
    st.session_state.setdefault("library", {}).pop(report_id, None)


def _queue(op: dict) -> None:
    op["op_id"] = uuid.uuid4().hex
    st.session_state.setdefault("library_ops", []).append(op)


def category_of(report: dict) -> str:
    return COMPARE_CATEGORY if report["kind"] == "compare" else report["meta"]["category"]


def display_title(report: dict) -> str:
    return report["meta"]["title"] or report["filename"]


SEARCH_SCOPES = {
    "전체": None,
    "제목·저자·키워드·태그": [],
} | {title: [key] for key, title, _ in SECTIONS}


def search_texts(report: dict, scope: str) -> list[str]:
    meta = report["meta"]
    head = [display_title(report), meta["authors"], meta["venue"], report["filename"],
            " ".join(meta["keywords"]), " ".join(report.get("tags", []))]
    keys = SEARCH_SCOPES[scope]
    if report["kind"] == "compare":
        body = [report["compare_text"]] if keys is None else []
    else:
        keys = [k for k, _, _ in SECTIONS] if keys is None else keys
        body = [report["sections"][k]["text"] or "" for k in keys]
    return body if SEARCH_SCOPES[scope] else head + body


def matches(report: dict, query: str, scope: str) -> str | None:
    """None if every word of the query isn't found; otherwise a short snippet around the first hit."""
    words = query.lower().split()
    texts = search_texts(report, scope)
    joined = "\n".join(texts).lower()
    if not all(w in joined for w in words):
        return None
    if not words:
        return ""
    for text in texts:
        index = text.lower().find(words[0])
        if index != -1:
            snippet = re.sub(r"\s+", " ", text[max(0, index - 30): index + 70]).strip()
            return f"…{snippet}…"
    return ""


def export_json(reports: dict) -> str:
    return json.dumps({"format": STORAGE_KEY, "reports": list(reports.values())}, ensure_ascii=False, indent=1)


def import_json(raw: bytes) -> int:
    """Queue every report in a backup file for saving. Returns how many were read."""
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != STORAGE_KEY:
        raise ValueError("이 앱에서 내려받은 보관함 백업 파일이 아닙니다.")
    reports = [r for r in payload.get("reports", []) if isinstance(r, dict) and r.get("id") and r.get("kind")]
    for report in reports:
        save(report)
    return len(reports)
