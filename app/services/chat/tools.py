import re
import json
import ast
from typing import Any, Dict, List, Optional, Tuple, Set

# --- Pseudo tool-call compatibility (e.g., OSS models emitting special markup) ---
PSEUDO_TOOL_RE = re.compile(
    r"<\|start\|>assistant<\|channel\|>commentary\s+to=(?:functions\.)?(\w+)(?:<\|channel\|>commentary\s+json|(?:\s+<\|constrain\|>json)?)\s*<\|message\|>(\{.*?\})<\|call\|>",
    re.DOTALL | re.IGNORECASE,
)

TOOL_NAME_MAPPING = {
    "perplexity_search": "perplexity_search",
    "functions.get_augmented_news": "get_augmented_news",
    "functions.get_company_profile": "get_company_profile",
    "functions.get_stock_quote": "get_stock_quote",
    "functions.get_historical_prices": "get_historical_prices",
    "functions.get_risk_assessment": "get_risk_assessment",
    "functions.rag_search": "rag_search",
    "functions.augmented_rag_search": "augmented_rag_search",
}

SUGGESTED_TOOL_WHITELIST = {
    "perplexity_search",
    "augmented_rag_search",
    "rag_search",
}

CODE_BLOCK_RE = re.compile(r"```(?:[\w+-]+)?\n(.*?)```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
SUGGESTED_CALL_RE = re.compile(r"(perplexity_search\s*\(.*?\))", re.DOTALL | re.IGNORECASE)
QUOTED_QUERY_RE = re.compile(r"[\"'“”『「](.*?)[\"'“”』」]")

MAX_SUGGESTED_TOOL_ROUNDS = 2

def normalize_tool_name_and_args(name: str, args_dict: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    mapped_name = TOOL_NAME_MAPPING.get(name, name)
    args = dict(args_dict or {})

    if "ticker" in args and "symbol" not in args:
        args["symbol"] = args.pop("ticker")

    if mapped_name in {"web_search", "perplexity_search"}:
        if "max_results" not in args:
            for alt_key in ("top_k", "top_n", "num_results", "limit"):
                if alt_key in args:
                    try:
                        args["max_results"] = int(args.pop(alt_key))
                    except (TypeError, ValueError):
                        args.pop(alt_key, None)
                    break

        if "recency_days" in args and "include_recent" not in args:
            recency_days = args.pop("recency_days")
            try:
                args["include_recent"] = int(recency_days) <= 7
            except (TypeError, ValueError):
                args["include_recent"] = True

        if "source" in args:
            source = args.pop("source")
            if source == "news":
                args["include_recent"] = True
                args["synthesize_answer"] = True
            elif source == "academic":
                args["include_recent"] = False
                args["synthesize_answer"] = True

    return mapped_name, args

def safe_literal_eval_node(node: ast.AST) -> Optional[Any]:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.Num):  # pragma: no cover - legacy AST nodes
        return node.n
    if isinstance(node, ast.NameConstant):  # pragma: no cover - legacy AST nodes
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = safe_literal_eval_node(node.operand)
        if operand is None:
            return None
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.List):
        items: List[Any] = []
        for elt in node.elts:
            value = safe_literal_eval_node(elt)
            if value is None:
                return None
            items.append(value)
        return items
    if isinstance(node, ast.Tuple):
        items: List[Any] = []
        for elt in node.elts:
            value = safe_literal_eval_node(elt)
            if value is None:
                return None
            items.append(value)
        return tuple(items)
    if isinstance(node, ast.Dict):
        result: Dict[Any, Any] = {}
        for key_node, value_node in zip(node.keys, node.values):
            key = safe_literal_eval_node(key_node)
            value = safe_literal_eval_node(value_node)
            if key is None or value is None:
                return None
            result[key] = value
        return result
    return None

def parse_suggested_tool_call(expr: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None

    call = getattr(tree, "body", None)
    if not isinstance(call, ast.Call):
        return None

    func = call.func
    if isinstance(func, ast.Attribute):
        func_name = func.attr
    elif isinstance(func, ast.Name):
        func_name = func.id
    else:
        return None

    func_name = func_name.strip()
    if func_name not in SUGGESTED_TOOL_WHITELIST:
        return None

    args_dict: Dict[str, Any] = {}

    if call.args:
        first_value = safe_literal_eval_node(call.args[0])
        if first_value is not None:
            if func_name in {"perplexity_search", "web_search", "augmented_rag_search", "rag_search"}:
                args_dict.setdefault("query", first_value)
            elif func_name.startswith("get_") and isinstance(first_value, str):
                args_dict.setdefault("symbol", first_value)

    for kw in call.keywords or []:
        if not kw.arg:
            continue
        value = safe_literal_eval_node(kw.value)
        if value is None:
            continue
        args_dict[kw.arg] = value

    mapped_name, normalized_args = normalize_tool_name_and_args(func_name, args_dict)
    return mapped_name, normalized_args

def extract_suggested_tool_calls(text: str, max_calls: int = 2) -> List[Dict[str, Any]]:
    if not text:
        return []

    candidates: List[str] = []

    for block_match in CODE_BLOCK_RE.finditer(text):
        block = block_match.group(1)
        if not block:
            continue
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        # Skip language specifiers (e.g., ```python)
        if len(lines) > 1 and re.match(r"^[A-Za-z][\w+-]*$", lines[0]) and "(" not in lines[0]:
            lines = lines[1:]
        for line in lines:
            candidates.append(line)

    for inline_match in INLINE_CODE_RE.finditer(text):
        snippet = inline_match.group(1).strip()
        if snippet:
            candidates.append(snippet)

    for call_match in SUGGESTED_CALL_RE.finditer(text):
        candidate = call_match.group(1).strip()
        if candidate:
            candidates.append(candidate)

    suggestions: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    counter = 0

    for candidate in candidates:
        if not any(name in candidate for name in SUGGESTED_TOOL_WHITELIST):
            continue
        parsed = parse_suggested_tool_call(candidate)
        if not parsed:
            continue
        mapped_name, normalized_args = parsed
        try:
            args_json = json.dumps(normalized_args)
        except Exception:
            continue
        key = (mapped_name, args_json)
        if key in seen:
            continue
        seen.add(key)
        counter += 1
        suggestions.append({
            "id": f"suggested-{counter}",
            "type": "function",
            "function": {"name": mapped_name, "arguments": args_json},
        })
        if len(suggestions) >= max_calls:
            break

    if len(suggestions) < max_calls:
        text_lower = text.lower()
        if "perplexity_search" in text_lower:
            quoted_queries: List[str] = []
            for match in QUOTED_QUERY_RE.findall(text):
                query = match.strip()
                if not query:
                    continue
                query = " ".join(part.strip() for part in query.splitlines() if part.strip())
                if len(query) < 3:
                    continue
                quoted_queries.append(query)

            if not quoted_queries:
                bullet_re = re.compile(r"^[\-\u2022\u30FB\u2219]\s*(.+)$")
                for line in text.splitlines():
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    bullet_match = bullet_re.match(line_stripped)
                    if bullet_match:
                        candidate_query = bullet_match.group(1).strip()
                        candidate_query = candidate_query.strip("\"'“”『』「」")
                        if len(candidate_query) >= 3:
                            quoted_queries.append(candidate_query)

            for query in quoted_queries:
                try:
                    normalized_query = query.strip()
                    if not normalized_query:
                        continue
                    mapped_name, normalized_args = normalize_tool_name_and_args(
                        "perplexity_search",
                        {
                            "query": normalized_query,
                            "synthesize_answer": True,
                            "include_recent": True,
                        },
                    )
                    args_json = json.dumps(normalized_args)
                    key = (mapped_name, args_json)
                    if key in seen:
                        continue
                    seen.add(key)
                    counter += 1
                    suggestions.append({
                        "id": f"suggested-{counter}",
                        "type": "function",
                        "function": {"name": mapped_name, "arguments": args_json},
                    })
                    if len(suggestions) >= max_calls:
                        break
                except Exception:
                    continue

    return suggestions

def extract_pseudo_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse pseudo tool calls embedded in assistant text into standard tool_call format."""
    calls: List[Dict[str, Any]] = []
    if not text:
        return calls
    try:
        matches = list(PSEUDO_TOOL_RE.finditer(text))
        counter = 0
        for m in matches:
            func_name = m.group(1)
            args_str = m.group(2)
            try:
                args = json.loads(args_str)
                mapped_name, normalized_args = normalize_tool_name_and_args(func_name, args)
                counter += 1
                calls.append({
                    "id": f"pseudo-{counter}",
                    "type": "function",
                    "function": {"name": mapped_name, "arguments": json.dumps(normalized_args)},
                })
            except Exception:
                continue
        return calls
    except Exception:
        return []

def strip_pseudo_tool_markup(text: str) -> str:
    """Remove pseudo tool-call markup blocks from assistant text for clean display."""
    if not text:
        return text
    try:
        return PSEUDO_TOOL_RE.sub("", text)
    except Exception:
        return text
