import pytest
import json
from app.services.chat.utils import (
    smart_truncate_answer,
    select_top_sources_for_context,
    build_web_search_tool_payload,
    safe_tool_result_json,
    TRUNC_MAX_CHARS
)
from app.services.chat.tools import (
    extract_pseudo_tool_calls,
    extract_suggested_tool_calls,
    normalize_tool_name_and_args,
    strip_pseudo_tool_markup
)
from app.services.chat.formatting import (
    human_preview_company_profile,
    human_preview_quote,
    human_preview_company_profile_jp,
    human_preview_quote_jp
)

def test_smart_truncate_answer():
    # Test short text
    assert smart_truncate_answer("Hello world") == "Hello world"
    
    # Test truncation
    long_text = "a" * (TRUNC_MAX_CHARS + 100)
    truncated = smart_truncate_answer(long_text)
    assert len(truncated) <= TRUNC_MAX_CHARS + 20 # allowing for "..."
    assert truncated.endswith("...")

def test_select_top_sources_for_context():
    sources = [
        {"url": "http://a.com", "content": "A"},
        {"url": "http://b.com", "content": "B"},
        {"url": "http://c.com", "content": "C"},
    ]
    # Test limit (argument is 'limit', not 'max_count')
    # The function expects a result dict containing 'sources', 'raw_sources', or 'results'
    result = {"sources": sources}
    top = select_top_sources_for_context(result, limit=2)
    assert len(top) == 2
    assert top[0]["url"] == "http://a.com"

def test_build_web_search_tool_payload():
    result = {
        "results": [
            {"url": "http://a.com", "content": "A", "title": "Title A"},
        ],
        "images": ["img1", "img2"]
    }
    payload = build_web_search_tool_payload(result)
    assert "top_sources" in payload
    assert len(payload["top_sources"]) == 1
    assert payload["top_sources"][0]["url"] == "http://a.com"
    assert "images" not in payload # Should be removed

def test_safe_tool_result_json():
    data = {"a": 1, "b": float('nan')}
    json_str = safe_tool_result_json(data)
    assert '"b": NaN' in json_str or '"b": null' in json_str or 'NaN' in json_str

def test_extract_pseudo_tool_calls():
    # Match the specific regex expected by the implementation
    text = 'Some text\n<|start|>assistant<|channel|>commentary to=get_stock_quote<|channel|>commentary json<|message|>{"symbol": "AAPL"}<|call|>'
    calls = extract_pseudo_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_stock_quote"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["symbol"] == "AAPL"

def test_strip_pseudo_tool_markup():
    text = 'Some text\n<|start|>assistant<|channel|>commentary to=get_stock_quote<|channel|>commentary json<|message|>{"symbol": "AAPL"}<|call|>'
    stripped = strip_pseudo_tool_markup(text)
    assert stripped.strip() == "Some text"

def test_extract_suggested_tool_calls():
    # Use a whitelisted tool
    text = 'Try running `perplexity_search("latest news on AAPL")`'
    calls = extract_suggested_tool_calls(text)
    assert len(calls) >= 1
    assert calls[0]["function"]["name"] == "perplexity_search"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["query"] == "latest news on AAPL"

def test_normalize_tool_name_and_args():
    name, args = normalize_tool_name_and_args("web_search", {"top_k": 5})
    assert name == "web_search"
    assert args["max_results"] == 5
    assert "top_k" not in args

def test_human_preview_company_profile():
    profile = {
        "symbol": "AAPL",
        "longName": "Apple Inc.",
        "sector": "Technology",
        "country": "US"
    }
    preview = human_preview_company_profile(profile)
    assert "Apple Inc." in preview
    assert "Technology" in preview
    assert "US" in preview

def test_human_preview_quote_jp():
    quote = {
        "symbol": "7203.T",
        "price": 2000,
        "currency": "JPY",
        "as_of": "2023-10-27"
    }
    preview = human_preview_quote_jp(quote)
    assert "7203.T" in preview
    assert "2,000円" in preview
