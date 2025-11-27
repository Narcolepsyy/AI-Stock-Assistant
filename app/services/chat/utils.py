import re
import json
from typing import Any, Dict, List
from urllib.parse import urlparse

# --- Smart truncation helpers for synthesized answers (Perplexity-style) ---
TRUNC_MAX_CHARS = 1500  # Target max visible chars for synthesized answers in tool payloads

WEB_SEARCH_METADATA_KEYS = ("confidence", "method", "timings", "search_time", "latency_ms")

PARTIAL_CITATION_RE = re.compile(r"\[[0-9]{0,3}$")  # Trailing partial like '[1' or '['
ORPHAN_NUM_RE = re.compile(r"(?<!\])(?:(?<=\s)|^)[0-9]{1,2}$")  # Trailing small number not closed by ]
BROKEN_CITATION_FRAGMENT_RE = re.compile(r"\[[0-9]{1,3}\s+[^\]]*$")  # e.g. '[1 Some partial'

# Pattern to detect raw JSON tool call outputs from GPT-5 (filter these out)
RAW_TOOL_CALL_JSON_RE = re.compile(
    r'\{\s*"query"\s*:\s*"[^"]*",\s*"max_results"\s*:\s*\d+,\s*"synthesize_answer"\s*:\s*true,\s*"include_recent"\s*:\s*true\s*\}',
    re.IGNORECASE | re.DOTALL
)
RAW_TOOL_CALL_FRAGMENT_RE = re.compile(
    r'"query"\s*:\s*"[^"]*"\s*,\s*"max_results"\s*:\s*\d+',
    re.IGNORECASE | re.DOTALL
)
REASONING_LINE_RE = re.compile(
    r"^\s*(?:reasoning|analysis|thought process|deliberation|scratchpad)[:：]\s*",
    re.IGNORECASE
)

def is_raw_tool_call_output(text: str) -> bool:
    """
    Detect if text contains raw JSON tool call output that should be hidden from user.
    GPT-5 sometimes outputs raw JSON instead of using proper function calling API.
    """
    if not text or len(text.strip()) < 10:
        return False
    
    # Check for JSON tool call patterns
    if RAW_TOOL_CALL_JSON_RE.search(text):
        return True

    # Catch partial/fragmented tool payloads that models (e.g., GPT-5) emit as reasoning text
    if RAW_TOOL_CALL_FRAGMENT_RE.search(text) and (
        '"synthesize_answer"' in text or '"include_recent"' in text
    ):
        return True
    
    # Check for multiple JSON objects in succession (tool call spam)
    json_brace_count = text.count('{')
    if json_brace_count >= 3 and '"query"' in text and '"max_results"' in text:
        return True
    
    return False

def smart_truncate_answer(answer: str, max_chars: int = TRUNC_MAX_CHARS) -> str:
    """Truncate synthesized answer safely, avoiding broken citation/artifacts.
    Rules:
    - Cut at max_chars boundary (soft) and roll back to last whitespace if mid-word
    - Remove dangling partial citation patterns (e.g. '[1', '[12')
    - Remove orphan trailing numbers (e.g. solitary '0' left after slice)
    - Preserve existing full citations like '[12]'
    - Append ellipsis if truncated
    """
    if not answer or len(answer) <= max_chars:
        return answer

    cut = answer[:max_chars]

    # If we cut in the middle of a word and we have sufficient earlier whitespace, roll back
    if not cut.endswith((" ", "\n", "\t")):
        last_space = cut.rfind(" ")
        if last_space > max_chars * 0.6:  # Only roll back if it preserves most content
            cut = cut[:last_space]

    # Remove trailing partial citation like '[1' or '['
    cut = PARTIAL_CITATION_RE.sub("", cut)

    # Remove broken citation fragments missing closing bracket
    cut = BROKEN_CITATION_FRAGMENT_RE.sub("", cut)

    # Remove trailing orphan number (e.g., stray '0') that is not part of [n]
    if ORPHAN_NUM_RE.search(cut.rstrip()):
        cut = ORPHAN_NUM_RE.sub("", cut.rstrip())

    # Clean dangling punctuation / unmatched opening brackets
    cut = cut.rstrip(" ,;:\n\t-[")

    return cut + "..."

def sanitize_perplexity_result(result: Any) -> Any:
    """Apply smart truncation & cleanup to perplexity_search tool result structure.
    Ensures no stray '0' or partial citation artifacts remain after truncation.
    """
    try:
        if not isinstance(result, dict):
            return result
        answer = result.get("answer")
        if isinstance(answer, str) and answer:
            truncated = smart_truncate_answer(answer, TRUNC_MAX_CHARS)
            # Remove any '[0]' citations (model sometimes enumerates from zero) – shift discouraged
            truncated = truncated.replace("[0]", "")
            # Remove verbose source markers that leak to the user
            truncated = re.sub(r"\s*\|\s*source=\w+\s*", " ", truncated).strip()
            # Remove lingering broken citation fragments inside (conservative: only if near end)
            tail = truncated[-120:]
            cleaned_tail = BROKEN_CITATION_FRAGMENT_RE.sub("", tail)
            if tail != cleaned_tail:
                truncated = truncated[:-120] + cleaned_tail
            result = {**result, "answer": truncated}
        return result
    except Exception:
        return result

def select_top_sources_for_context(result: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    """Extract a compact list of sources for model context while preserving citation ids."""
    candidates: List[Dict[str, Any]] = []
    for key in ("sources", "raw_sources", "results"):
        value = result.get(key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))

    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for item in candidates:
        citation_id = item.get("citation_id") or item.get("citationId") or item.get("id")
        url = item.get("url") or item.get("link") or item.get("source")
        title = item.get("title") or item.get("name")
        dedupe_key = str(citation_id).strip() if citation_id is not None else (url or title)
        if not dedupe_key:
            continue
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        summary = {
            "citation_id": citation_id,
            "title": title,
            "url": url,
        }

        snippet = item.get("snippet") or item.get("description") or item.get("content") or item.get("text")
        if snippet:
            summary["snippet"] = snippet

        domain = item.get("domain")
        if not domain and isinstance(url, str):
            try:
                parsed = urlparse(url if re.match(r"^https?://", url, re.IGNORECASE) else f"https://{url}")
                domain = parsed.hostname or parsed.path.split("/")[0]
                if domain and domain.startswith("www."):
                    domain = domain[4:]
            except Exception:
                domain = None
        if domain:
            summary["domain"] = domain

        publish_date = (
            item.get("publish_date")
            or item.get("publishDate")
            or item.get("timestamp")
            or item.get("date")
        )
        if publish_date:
            summary["publish_date"] = publish_date

        score = item.get("score") or item.get("relevance_score")
        if score is not None:
            summary["score"] = score

        selected.append(summary)
        if len(selected) >= limit:
            break

    return selected

def build_web_search_tool_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Create a compact payload for web search tools that preserves inline citations."""
    payload: Dict[str, Any] = {}

    answer = result.get("answer")
    if isinstance(answer, str) and answer:
        payload["answer"] = smart_truncate_answer(answer, TRUNC_MAX_CHARS)

    citations = result.get("citations")
    if isinstance(citations, dict) and citations:
        payload["citations"] = citations

    # Preserve citation styling if available
    for key in ("css", "styles", "style_tag"):
        value = result.get(key)
        if value:
            payload[key] = value
            break

    metadata = {}
    for key in WEB_SEARCH_METADATA_KEYS:
        value = result.get(key)
        if value is not None:
            metadata[key] = value
    synthesized_query = result.get("synthesized_query") or result.get("query")
    if synthesized_query:
        metadata["query"] = synthesized_query
    if metadata:
        payload["metadata"] = metadata

    sources = select_top_sources_for_context(result)
    if sources:
        payload["top_sources"] = sources

    return payload

def normalize_content_piece(piece: Any) -> str:
    """Extract text from various content shapes safely."""
    try:
        if piece is None:
            return ""
        if isinstance(piece, str):
            return _strip_reasoning_text(piece)
        # Content piece could be dict-like {type: 'text', 'text': '...'}
        if isinstance(piece, dict):
            piece_type = str(piece.get("type") or "").lower()
            # Hide explicit reasoning chunks emitted by reasoning-capable models (e.g., GPT-5)
            if piece_type in {"reasoning", "analysis", "reflection"}:
                return ""
            txt = piece.get("text") or piece.get("content") or ""
            normalized = txt if isinstance(txt, str) else json.dumps(txt)[:2000]
            return _strip_reasoning_text(normalized)
        # Fallback stringify
        return _strip_reasoning_text(str(piece))
    except Exception:
        return ""

def _strip_reasoning_text(text: str) -> str:
    """Remove visible reasoning lines/prefixes emitted by reasoning models."""
    if not text:
        return ""
    filtered_lines = []
    for line in text.splitlines():
        if REASONING_LINE_RE.match(line):
            continue
        if _looks_like_tool_preamble(line):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)


def _looks_like_tool_preamble(line: str) -> bool:
    """Detect planning-style lines (e.g., GPT-5 reasoning) that describe upcoming tool calls."""
    lowered = line.strip().lower()
    if lowered.startswith(("searching for", "looking for", "looking up", "gathering", "fetching")):
        if "{" in line or '"query"' in line or "max_results" in line or "synthesize_answer" in line:
            return True
    return False

def safe_tool_result_json(payload: Any) -> str:
    """Serialize tool payloads without raising on non-JSON types."""
    try:
        return json.dumps(payload, default=str)
    except Exception:
        try:
            return json.dumps(str(payload))
        except Exception:
            return "\"\""
