import base64
import re
import time
from typing import Callable

import anthropic
from google import genai
from google.genai import _interactions_errors as gemini_errors  # Interactions API errors aren't re-exported publicly
from google.genai import types as gemini_types

from pdf_processor import chunk_text, split_references

# Prices are USD per 1M tokens, used only for the rough cost shown after an analysis.
MODELS = {
    "claude-opus-5-5": {
        "label": "Claude Opus 5.5 (가장 정확, 비쌈)",
        "input": 4.00, "output": 20.00, "cache_read": 0.20, "fallback": True,
    },
    "claude-sonnet-5-5": {
        "label": "Claude Sonnet 5.5 (중간)",
        "input": 2.00, "output": 10.00, "cache_read": 0.10, "fallback": True,
    },
    "claude-haiku-5-5": {
        "label": "Claude Haiku 5.5 (가장 저렴)",
        "input": 0.10, "output": 0.50, "cache_read": 0.01, "fallback": False,
    },
}

EFFORT = "medium"
MAX_DIRECT_TOKENS = 100_000
CHUNK_TOKENS = 60_000

# Both are free-tier models; the second is used when the first is overloaded, slow, or out of quota.
# On the free tier flash-lite answered in seconds while gemini-3.8-flash (and gemini-4-hydrogen) took minutes.
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
GEMINI_MAX_DIRECT_CHARS = 300_000
GEMINI_CHUNK_CHARS = 120_000
GEMINI_TIMEOUT_SECONDS = 180
GEMINI_MAX_OUTPUT_TOKENS = 32_000

SYSTEM_PROMPT = """당신은 스포츠 경영·스포츠 산업·팬 문화·스포츠 통계 분야의 학술 논문을 분석하는 연구 조교입니다. 사용자는 이 분야의 대학원생입니다.

- 논문이 한국어든 영어든 답변은 반드시 한국어로 쓰세요. 전문용어는 처음 나올 때 괄호 안에 원어를 함께 쓰세요. 예: 구조방정식 모형(SEM)
- 논문에 있는 내용만 근거로 쓰세요. 논문에서 확인할 수 없는 항목은 지어내지 말고 "논문에서 확인되지 않음"이라고 쓰세요.
- 저자가 직접 밝힌 내용과 당신의 해석을 구분하고, 해석에는 "(분석자 해석)"이라고 표시하세요.
- 논문이 PDF에서 추출한 텍스트로 주어지면 줄바꿈, 단 배치, 표가 깨져 있을 수 있습니다. 문맥으로 판단하세요.
- 요청받은 형식의 마크다운만 출력하세요. 인사말이나 맺음말은 쓰지 마세요."""

SECTIONS = [
    ("summary", "요약/핵심 주장", """이 논문의 요약을 다음 형식으로 쓰세요.

### 기본 정보
- 제목:
- 저자:
- 발행 연도 / 게재지:

### 요약
연구 목적, 핵심 연구질문(또는 가설), 주요 발견, 결론을 5~8문장의 한 문단으로 요약하세요.

### 핵심 주장
저자가 가장 강조하는 주장을 3개 이내의 글머리표로 정리하세요."""),

    ("methods", "연구방법·통계기법", """이 논문의 연구방법을 다음 형식으로 구체적으로 정리하세요.

### 연구설계
양적/질적/혼합 연구 중 무엇인지와 그렇게 판단한 근거.

### 표본
대상, 표본 수(최종 분석 표본 포함), 표집 방법, 자료 수집 시기와 방법.

### 측정도구
변수별로 척도, 출처(원척도 저자), 문항 수, 응답 방식, 신뢰도·타당도 지표(수치 포함).

### 통계기법
분석 순서대로 다음 표를 채우세요.
| 분석 단계 | 사용 기법 | 목적(검증한 가설·질문) | 주요 결과 |
표 아래에 사용 소프트웨어(SPSS, AMOS, R 등)를 쓰세요. 매개·조절효과 분석이 있으면 검증 방법(부트스트래핑, PROCESS 모형 번호 등)을 구체적으로 쓰세요."""),

    ("references", "인용·참고문헌", """이 논문의 인용과 참고문헌을 다음 형식으로 정리하세요.

### 핵심 선행연구
본문에서 자주 또는 중요하게 인용된 선행연구 5~10개를 중요도 순으로 고르세요.
- **저자 (연도)** — 그 연구의 핵심 내용 / 이 논문에서 어떤 역할로 인용되었는지(이론적 근거, 측정도구 출처, 비교 대상 등)

### 참고문헌 목록
논문 끝의 참고문헌 목록을 원문 그대로, 번역하지 말고 번호를 붙여 나열하세요. 목록을 찾을 수 없거나 텍스트가 깨져 일부만 읽힌다면 그 사실을 먼저 밝히고, 읽을 수 있는 항목만 나열하세요."""),

    ("gap", "차별점·갭 분석", """이 논문을 선행연구와 비교해 다음 형식으로 분석하세요.

### 이론적 차별점

### 방법론적 차별점

위 두 항목에서는 저자가 지적한 선행연구의 한계(연구 갭)와, 이 논문이 그것을 어떻게 메우는지를 연결해서 쓰세요.

### 저자가 밝힌 연구의 한계

### 후속 연구 제언

위 두 항목은 저자가 직접 쓴 내용만 정리하세요.

### 추가로 볼 만한 연구 갭 (분석자 해석)
저자가 언급하지 않았지만 이 논문을 바탕으로 이어갈 수 있는 연구 방향을 2~3개 제안하세요."""),
]

CATEGORIES = [
    "스포츠 마케팅·스폰서십",
    "팬·관중 행동/문화",
    "스포츠 산업·경제",
    "스포츠 경영·조직·정책",
    "스포츠 데이터·통계 분석",
    "스포츠 미디어·커뮤니케이션",
    "기타",
]
DESIGNS = ["양적", "질적", "혼합", "기타"]

META_INSTRUCTION = f"""이 논문의 서지 정보와 분류를 아래 형식 그대로, 한 줄에 한 항목씩 쓰세요. 다른 말은 쓰지 마세요. 알 수 없는 항목은 비워 두세요.
제목: (논문 원제목 그대로, 번역하지 말 것)
저자: (쉼표로 구분)
연도: (4자리 숫자)
게재지:
분야: (다음 중 가장 가까운 하나를 그대로: {" / ".join(CATEGORIES)})
연구유형: (다음 중 하나: {" / ".join(DESIGNS)})
키워드: (논문의 핵심 주제어 3~6개, 한국어, 쉼표로 구분)"""

COMPARE_SYSTEM_PROMPT = """당신은 스포츠 경영·스포츠 산업·팬 문화·스포츠 통계 분야의 학술 논문을 비교 분석하는 연구 조교입니다. 사용자는 이 분야의 대학원생입니다.

- 답변은 반드시 한국어로 쓰세요.
- 주어진 논문 분석 리포트에 있는 내용만 근거로 쓰세요. 리포트에서 확인할 수 없는 항목은 지어내지 말고 "확인되지 않음"이라고 쓰세요.
- 리포트 내용과 당신의 해석을 구분하고, 해석에는 "(분석자 해석)"이라고 표시하세요.
- 요청받은 형식의 마크다운만 출력하세요. 인사말이나 맺음말은 쓰지 마세요."""

COMPARE_INSTRUCTION = """위 {n}편의 논문 분석 리포트를 바탕으로 논문들을 비교하는 리포트를 다음 형식으로 쓰세요. 논문은 "논문 1", "논문 2"처럼 번호로 부르세요.

### 비교한 논문
번호별로 저자(연도), 제목을 한 줄씩 쓰세요.

### 한눈에 비교
다음 행을 가진 표를 만드세요. 열은 "항목"과 논문 번호입니다.
연구 주제·목적 / 이론적 배경 / 연구설계 / 표본(대상·표본 수) / 핵심 변수 / 주요 통계기법 / 주요 결과 / 저자가 밝힌 한계

### 공통점
연구 주제, 이론, 방법, 결과에서 겹치는 점을 글머리표로 정리하세요.

### 차이점
연구 관점·이론, 표본과 방법론, 결과를 나눠서 구체적으로 비교하세요.

### 결과의 연결 (분석자 해석)
논문들의 결과가 서로 보완하는지, 충돌하는지 설명하세요. 충돌한다면 표본·측정·분석 방법의 차이로 설명할 수 있는지 쓰세요.

### 함께 읽었을 때 보이는 연구 갭과 후속 연구 아이디어 (분석자 해석)
논문들을 함께 놓고 볼 때 드러나는 연구 갭과, 이를 메울 수 있는 후속 연구 아이디어 2~3개를 연구질문 형태로 제안하세요."""

CONDENSE_PROMPT = """아래는 긴 논문을 {n}개로 나눈 것 중 {i}번째 부분입니다. 나중에 이 노트만 보고 논문 전체를 분석할 수 있도록, 이 부분을 빠짐없이 압축한 노트를 쓰세요.

다음 정보는 수치와 고유명사까지 그대로 남기세요: 연구 목적·연구질문·가설, 표본과 자료 수집, 측정도구, 통계기법과 결과 수치, 인용된 선행연구(저자, 연도, 인용 맥락), 저자가 밝힌 한계와 후속 연구 제언.

<part>
{chunk}
</part>"""

# Plain-text sections split by marker lines, so a partly malformed answer still yields the sections it has.
GEMINI_PROMPT = (
    "아래 다섯 항목을 순서대로 모두 작성하세요. 각 항목은 반드시 구분자 한 줄(예: <<<summary>>>)로 시작하고, "
    "그다음 줄부터 지시된 형식으로 쓰세요. 구분자 줄에는 다른 글자를 쓰지 마세요.\n\n"
    + "\n\n".join(f"<<<{key}>>> ({title})\n{instruction}" for key, title, instruction in SECTIONS)
    + f"\n\n<<<meta>>> (서지 정보·분류)\n{META_INSTRUCTION}"
)
GEMINI_MARKER = re.compile(
    r"^\s*<<<(" + "|".join([key for key, _, _ in SECTIONS] + ["meta"]) + r")>>>.*$", re.MULTILINE
)

ProgressCallback = Callable[[str, float], None]
API_ERRORS = (anthropic.APIError, gemini_errors.APIError)


class AnalysisError(Exception):
    pass


class RefusalError(AnalysisError):
    pass


def friendly_error(e: Exception) -> str:
    if isinstance(e, gemini_errors.APIError):
        return _friendly_gemini_error(e)
    if isinstance(e, anthropic.AuthenticationError):
        return "API 키가 올바르지 않습니다. 키를 다시 확인해 주세요."
    if isinstance(e, anthropic.PermissionDeniedError):
        return f"이 API 키로는 요청할 수 없습니다: {e.message}"
    if isinstance(e, anthropic.NotFoundError):
        return "선택한 모델을 이 계정에서 사용할 수 없습니다. 다른 모델을 골라 보세요."
    if isinstance(e, anthropic.BadRequestError):
        return f"요청이 거부되었습니다 (크레딧 잔액 부족일 수 있습니다): {e.message}"
    if isinstance(e, anthropic.RateLimitError):
        return "요청 한도를 넘었습니다. 1~2분 뒤 다시 시도해 주세요."
    if isinstance(e, anthropic.APIConnectionError):
        return "Claude 서버에 연결하지 못했습니다. 인터넷 연결을 확인해 주세요."
    if isinstance(e, anthropic.APIStatusError):
        return f"Claude API 오류 ({e.status_code}): {e.message}"
    return f"Claude API 오류: {e}"


def _friendly_gemini_error(e: gemini_errors.APIError) -> str:
    if isinstance(e, gemini_errors.RateLimitError):
        return (
            "무료 모드 사용 한도를 넘었습니다. 1분 뒤 다시 시도해 보고, 계속 안 되면 오늘 한도가 다 찬 것입니다"
            "(한국 시간 오후 4~5시에 초기화). 본인 Claude API 키를 넣으면 바로 분석할 수 있습니다."
        )
    if isinstance(e, (gemini_errors.AuthenticationError, gemini_errors.PermissionDeniedError)) or (
        isinstance(e, gemini_errors.BadRequestError) and "API key" in str(e)
    ):
        return "무료 모드 설정(서버의 Gemini 키)에 문제가 있습니다. 앱 관리자에게 알려 주세요."
    if isinstance(e, gemini_errors.APITimeoutError):
        return "무료 모델의 응답이 너무 오래 걸려 중단했습니다. 잠시 뒤 다시 시도해 주세요."
    if isinstance(e, gemini_errors.APIConnectionError):
        return "Google 서버에 연결하지 못했습니다. 잠시 뒤 다시 시도해 주세요."
    if isinstance(e, gemini_errors.APIStatusError) and e.status_code >= 500:
        return "Google 무료 모델이 지금 붐벼서 분석하지 못했습니다. 몇 분 뒤 다시 시도해 주세요."
    if isinstance(e, gemini_errors.APIStatusError):
        return f"Gemini API 오류 ({e.status_code}). 잠시 뒤 다시 시도해 주세요."
    return f"Gemini API 오류: {e}"


def parse_meta(text: str | None) -> dict:
    """Parse META_INSTRUCTION's "항목: 값" lines; missing or invalid values become defaults."""
    fields = {"제목": "title", "저자": "authors", "연도": "year", "게재지": "venue",
              "분야": "category", "연구유형": "design", "키워드": "keywords"}
    meta = {"title": "", "authors": "", "year": "", "venue": "", "category": "기타", "design": "기타", "keywords": []}
    for line in (text or "").splitlines():
        label, sep, value = line.strip().lstrip("-*").strip().partition(":")
        key = fields.get(label.strip().replace("*", ""))
        if not sep or not key:
            continue
        value = value.strip().strip("*").strip()
        if value.startswith("(") and value.endswith(")"):
            continue  # the model echoed the placeholder
        if key == "keywords":
            meta[key] = [k.strip() for k in re.split(r"[,，、]", value) if k.strip()][:8]
        elif key == "year":
            found = re.search(r"(19|20)\d{2}", value)
            meta[key] = found.group(0) if found else ""
        elif key == "category":
            meta[key] = next((c for c in CATEGORIES if c.replace(" ", "") in value.replace(" ", "")), "기타")
        elif key == "design":
            meta[key] = next((d for d in DESIGNS if d in value), "기타")
        else:
            meta[key] = value
    return meta


def compare_document(reports: list[dict]) -> str:
    """The analyses of each paper, used as the comparison's source instead of the full papers."""
    parts = []
    for n, report in enumerate(reports, 1):
        meta = report["meta"]
        lines = [f'<paper number="{n}">', f"제목: {meta['title'] or report['filename']}",
                 f"저자: {meta['authors']}", f"연도: {meta['year']}"]
        for key, title, _ in SECTIONS:
            text = report["sections"][key]["text"]
            if not text:
                continue
            if key == "references":
                text = text.split("### 참고문헌 목록")[0].strip()  # the raw list adds length, not insight
            lines += ["", f"## {title}", text]
        lines.append("</paper>")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _condense(text: str, chunk_chars: int, ask: Callable[[str], str], on_progress: ProgressCallback) -> str:
    body, references = split_references(text)
    chunks = chunk_text(body, chunk_chars)
    notes = []
    for i, chunk in enumerate(chunks, 1):
        on_progress(f"긴 논문이라 나눠서 읽는 중... ({i}/{len(chunks)})", 0.02 + 0.18 * i / len(chunks))
        notes.append(f"## 부분 {i} 노트\n{ask(CONDENSE_PROMPT.format(n=len(chunks), i=i, chunk=chunk))}")

    document = "[분량이 많아 본문을 부분별로 압축한 노트입니다]\n\n" + "\n\n".join(notes)
    if references:
        document += f"\n\n[참고문헌 원문]\n{references}"
    return document


class ClaudeAnalyzer:
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.label = MODELS[model]["label"]
        self.usage = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
        self.chunked = False

    def analyze(self, text: str, on_progress: ProgressCallback, pdf_bytes: bytes | None = None) -> tuple[dict, dict]:
        """Return (sections, meta). With pdf_bytes the model reads the PDF itself (scanned or garbled text)."""
        # Errors that would fail every section (bad key, no credit, offline) propagate.
        if pdf_bytes:
            paper_block = {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf",
                           "data": base64.standard_b64encode(pdf_bytes).decode()},
                "cache_control": {"type": "ephemeral"},
            }
        else:
            on_progress("논문 길이를 확인하는 중...", 0.02)
            document = self._prepare_document(text, on_progress)
            paper_block = {
                "type": "text",
                "text": f"<paper>\n{document}\n</paper>",
                "cache_control": {"type": "ephemeral"},
            }

        results = {}
        for i, (key, title, instruction) in enumerate(SECTIONS):
            on_progress(f"{i + 1}/{len(SECTIONS)} {title} 분석 중...", 0.2 + 0.8 * i / len(SECTIONS))
            try:
                answer = self._ask(SYSTEM_PROMPT, [paper_block, {"type": "text", "text": instruction}])
                results[key] = {"text": answer, "error": None}
            except RefusalError as e:
                results[key] = {"text": None, "error": str(e)}
            except anthropic.APIStatusError as e:
                if e.status_code != 429 and e.status_code < 500:
                    raise
                results[key] = {"text": None, "error": friendly_error(e)}

        on_progress("분야·키워드를 정리하는 중...", 0.97)
        try:
            meta = parse_meta(self._ask(SYSTEM_PROMPT, [paper_block, {"type": "text", "text": META_INSTRUCTION}]))
        except (RefusalError, anthropic.APIStatusError):
            meta = parse_meta(None)  # classification is optional; the analysis itself succeeded
        on_progress("완료", 1.0)
        return results, meta

    def compare(self, reports: list[dict], on_progress: ProgressCallback) -> str:
        on_progress(f"논문 {len(reports)}편을 비교하는 중...", 0.1)
        prompt = f"{compare_document(reports)}\n\n{COMPARE_INSTRUCTION.format(n=len(reports))}"
        answer = self._ask(COMPARE_SYSTEM_PROMPT, prompt)
        on_progress("완료", 1.0)
        return answer

    def estimated_cost(self) -> float:
        p = MODELS[self.model]
        u = self.usage
        return (
            u["input"] * p["input"]
            + u["cache_write"] * p["input"] * 1.25
            + u["cache_read"] * p["cache_read"]
            + u["output"] * p["output"]
        ) / 1_000_000

    def _prepare_document(self, text: str, on_progress: ProgressCallback) -> str:
        tokens = self.client.messages.count_tokens(
            model=self.model,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        ).input_tokens
        if tokens <= MAX_DIRECT_TOKENS:
            return text
        self.chunked = True
        chunk_chars = int(CHUNK_TOKENS * len(text) / tokens)
        return _condense(text, chunk_chars, lambda prompt: self._ask(SYSTEM_PROMPT, prompt), on_progress)

    def _ask(self, system: str, content) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": 64000,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": EFFORT},
        }
        if MODELS[self.model]["fallback"]:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"

        with self.client.beta.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()

        u = message.usage
        self.usage["input"] += u.input_tokens or 0
        self.usage["output"] += u.output_tokens or 0
        self.usage["cache_write"] += u.cache_creation_input_tokens or 0
        self.usage["cache_read"] += u.cache_read_input_tokens or 0

        if message.stop_reason == "refusal":
            raise RefusalError("모델이 이 항목에 대한 답변을 거절했습니다.")
        answer = "".join(b.text for b in message.content if b.type == "text").strip()
        if message.stop_reason == "max_tokens":
            answer += "\n\n> 응답이 길이 제한에 걸려 중간에 잘렸습니다."
        return answer


class GeminiAnalyzer:
    # One request for all four sections, to stay within the free tier's request and token quotas.

    def __init__(self, api_key: str):
        retries = gemini_types.HttpRetryOptions(attempts=2)
        self.client = genai.Client(api_key=api_key, http_options=gemini_types.HttpOptions(retry_options=retries))
        self.model = GEMINI_MODELS[0]
        self.usage = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
        self.chunked = False

    @property
    def label(self) -> str:
        return f"Google Gemini 무료 모드 ({self.model})"

    def analyze(self, text: str, on_progress: ProgressCallback, pdf_bytes: bytes | None = None) -> tuple[dict, dict]:
        """Return (sections, meta). With pdf_bytes the model reads the PDF itself (scanned or garbled text)."""
        if pdf_bytes:
            prompt = [
                {"type": "document", "data": base64.standard_b64encode(pdf_bytes).decode(),
                 "mime_type": "application/pdf"},
                {"type": "text", "text": GEMINI_PROMPT},
            ]
        else:
            on_progress("논문 길이를 확인하는 중...", 0.02)
            document = text
            if len(text) > GEMINI_MAX_DIRECT_CHARS:
                self.chunked = True
                document = _condense(text, GEMINI_CHUNK_CHARS, self._ask, on_progress)
            prompt = f"<paper>\n{document}\n</paper>\n\n{GEMINI_PROMPT}"

        on_progress("네 항목을 한꺼번에 분석하는 중... (보통 1~3분)", 0.3)
        answer = self._ask(prompt)
        parts = GEMINI_MARKER.split(answer)
        found = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}

        results = {}
        for key, _, _ in SECTIONS:
            if found.get(key):
                results[key] = {"text": found[key], "error": None}
            else:
                results[key] = {"text": None, "error": "이 항목의 결과를 받지 못했습니다. 다시 분석해 보세요."}
        on_progress("완료", 1.0)
        return results, parse_meta(found.get("meta"))

    def compare(self, reports: list[dict], on_progress: ProgressCallback) -> str:
        on_progress(f"논문 {len(reports)}편을 비교하는 중... (보통 1분 안팎)", 0.1)
        prompt = f"{compare_document(reports)}\n\n{COMPARE_INSTRUCTION.format(n=len(reports))}"
        answer = self._ask(prompt, system=COMPARE_SYSTEM_PROMPT)
        on_progress("완료", 1.0)
        return answer

    def estimated_cost(self) -> float:
        return 0.0

    def _ask(self, prompt: str | list, system: str = SYSTEM_PROMPT) -> str:
        candidates = GEMINI_MODELS[GEMINI_MODELS.index(self.model):]
        for i, model in enumerate(candidates):
            started = time.monotonic()
            try:
                interaction = self.client.interactions.create(
                    model=model,
                    input=prompt,
                    system_instruction=system,
                    generation_config={"max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS},
                    timeout=GEMINI_TIMEOUT_SECONDS,
                )
                self.model = model
                break
            except (gemini_errors.APIStatusError, gemini_errors.APITimeoutError) as e:
                timed_out = isinstance(e, gemini_errors.APITimeoutError)
                code = "timeout" if timed_out else e.status_code
                print(f"[gemini] model={model} error={code} seconds={time.monotonic() - started:.0f}", flush=True)
                retryable = timed_out or code == 429 or code >= 500
                if not retryable or i == len(candidates) - 1:
                    raise

        u = interaction.usage
        print(
            f"[gemini] model={model} status={interaction.status} seconds={time.monotonic() - started:.0f} "
            f"input={u.total_input_tokens if u else None} output={u.total_output_tokens if u else None} "
            f"thought={u.total_thought_tokens if u else None}",
            flush=True,
        )
        if u:
            self.usage["input"] += u.total_input_tokens or 0
            self.usage["output"] += u.total_output_tokens or 0
        if interaction.status == "incomplete":
            raise AnalysisError("응답이 길이 제한에 걸려 잘렸습니다. 다시 분석해 보세요.")
        if interaction.status != "completed":
            raise AnalysisError(f"Gemini가 분석을 끝내지 못했습니다 (상태: {interaction.status}). 다시 시도해 주세요.")
        return (interaction.output_text or "").strip()
