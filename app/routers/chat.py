"""Chat API routes for conversational AI functionality."""
import ast
import json
import logging
import re
import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple, Set
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from openai import AsyncAzureOpenAI

from app.models.database import get_db, User, Log
from app.models.schemas import ChatRequest, ChatResponse, ChatHistoryResponse, ChatMessage
from app.auth.dependencies import get_current_user
from app.core.config import DEFAULT_MODEL, ML_TOOL_SELECTION_ENABLED
from app.utils.conversation import (
    conv_clear, conv_get, estimate_tokens, MAX_TOKENS_PER_TURN,
    prepare_conversation_messages_with_memory, store_conversation_messages_with_memory
)
from app.utils.tools import TOOL_REGISTRY, build_tools_for_request, build_tools_for_request_ml
from app.utils.connection_pool import connection_pool
from app.utils.request_cache import (
    get_cached_response, cache_response, is_request_in_flight,
    mark_request_in_flight, clear_request_in_flight
)
from app.utils.query_optimizer import (
    is_simple_query, get_fast_model_recommendation, should_skip_rag_and_web_search
)
from app.utils.tool_usage_logger import log_tool_usage
from app.services.chat.utils import (
    is_raw_tool_call_output, smart_truncate_answer, sanitize_perplexity_result,
    select_top_sources_for_context, build_web_search_tool_payload,
    normalize_content_piece, safe_tool_result_json,
    TRUNC_MAX_CHARS, WEB_SEARCH_METADATA_KEYS
)
from app.services.chat.tools import (
    extract_pseudo_tool_calls, extract_suggested_tool_calls,
    normalize_tool_name_and_args, PSEUDO_TOOL_RE,
    MAX_SUGGESTED_TOOL_ROUNDS, SUGGESTED_TOOL_WHITELIST,
    strip_pseudo_tool_markup
)
from app.services.chat.formatting import (
    human_preview_company_profile, human_preview_quote,
    human_preview_company_profile_jp, human_preview_quote_jp,
    human_preview_historical_jp, build_news_summary,
    human_preview_from_summary, human_preview_historical,
    human_preview_nikkei_news_jp
)


router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

# Optimized Japanese directive - shorter and more efficient
_JAPANESE_DIRECTIVE = "\n\n日本語で回答してください。"

# Thread pool for async tool execution
_tool_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool-exec")

# Pre-serialized common response structures to reduce JSON overhead
_PRECOMPILED_RESPONSES = {
    'start': lambda conv_id, model: f'{{"type":"start","conversation_id":"{conv_id}","model":"{model}"}}',
    'tool_running': lambda name: f'{{"type":"tool_call","name":"{name}","status":"running"}}',
    'tool_completed': lambda name: f'{{"type":"tool_call","name":"{name}","status":"completed"}}',
    'tool_error': lambda name, error: f'{{"type":"tool_call","name":"{name}","status":"error","error":{json.dumps(str(error))}}}',
    'content': lambda delta: f'{{"type":"content","delta":{json.dumps(delta)}}}'
}

# --- Smart truncation helpers for synthesized answers (Perplexity-style) ---
# TRUNC_MAX_CHARS moved to app.services.chat.utils

_WEB_SEARCH_TOOL_NAMES = {
    "perplexity_search",
    "web_search",
    "web_search_news",
    "augmented_rag_search",
    "financial_context_search",
    "augmented_rag_web",
}

# _WEB_SEARCH_METADATA_KEYS moved to app.services.chat.utils

# Regex patterns moved to app.services.chat.utils and app.services.chat.tools

# --- Pseudo tool-call compatibility (e.g., OSS models emitting special markup) ---
_PSEUDO_TOOL_RE = re.compile(
    r"<\|start\|>assistant<\|channel\|>commentary\s+to=(?:functions\.)?(\w+)(?:<\|channel\|>commentary\s+json|(?:\s+<\|constrain\|>json)?)\s*<\|message\|>(\{.*?\})<\|call\|>",
    re.DOTALL | re.IGNORECASE,
)

_TOOL_NAME_MAPPING = {
    "perplexity_search": "perplexity_search",
    "functions.get_augmented_news": "get_augmented_news",
    "functions.get_company_profile": "get_company_profile",
    "functions.get_stock_quote": "get_stock_quote",
    "functions.get_historical_prices": "get_historical_prices",
    "functions.get_risk_assessment": "get_risk_assessment",
    "functions.rag_search": "rag_search",
    "functions.augmented_rag_search": "augmented_rag_search",
}

_SUGGESTED_TOOL_WHITELIST = {
    "perplexity_search",
    "augmented_rag_search",
    "rag_search",
}

_CODE_BLOCK_RE = re.compile(r"```(?:[\w+-]+)?\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_SUGGESTED_CALL_RE = re.compile(r"(perplexity_search\s*\(.*?\))", re.DOTALL | re.IGNORECASE)
_QUOTED_QUERY_RE = re.compile(r"[\"'“”『「](.*?)[\"'“”』」]")

MAX_SUGGESTED_TOOL_ROUNDS = 2




def _convert_latex_format(text: str) -> str:
    r"""Normalize LaTeX math delimiters to `\(`, `\)`, `\[`, and `\]` as required by frontend rendering."""
    if not text:
        return text
    try:
        # 1. Convert block math expressed with $$ ... $$ to \[ ... \]
        def _replace_block(match: re.Match[str]) -> str:
            inner = match.group(1).strip()
            return f'\\[{inner}\\]'

        converted = re.sub(r'\$\$(.+?)\$\$', _replace_block, text, flags=re.DOTALL)

        # 2. Convert inline math expressed with single $ ... $ to \( ... \)
        def _replace_inline(match: re.Match[str]) -> str:
            inner = match.group(1).strip()
            return f'\\({inner}\\)'

        converted = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', _replace_inline, converted, flags=re.DOTALL)

        # 3. Convert custom bracket style [ \command ... ] to inline math
        def _replace_bracket(match: re.Match[str]) -> str:
            content = match.group(0)
            inner = content[1:-1].strip()
            return f'\\({inner}\\)'

        converted = re.sub(r'\[\s*\\[a-zA-Z]+.*?\]', _replace_bracket, converted, flags=re.DOTALL)

        return converted
    except Exception as e:
        logger.warning(f"Failed to convert LaTeX format: {e}")
        return text

@router.get("/models")
@router.head("/models")
def list_models():
    """List available AI models with enhanced selection support."""
    from app.services.openai_client import get_available_models

    available_models = get_available_models()

    # Format for frontend consumption
    models_list = []
    for model_key, config in available_models.items():
        models_list.append({
            "id": model_key,
            "name": config["display_name"],
            "description": config["description"],
            "provider": config["provider"],
            "available": config["available"],
            "default": config.get("default", False)
        })

    # Sort by availability first, then by default, then alphabetically
    models_list.sort(key=lambda x: (not x["available"], not x.get("default", False), x["name"]))

    return {
        "default": DEFAULT_MODEL,
        "available": models_list,
        "total_count": len(models_list),
        "available_count": len([m for m in models_list if m["available"]])
    }

@router.post("/stream")
async def chat_stream_endpoint(
    req: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Streaming chat endpoint with real-time response streaming, caching, and performance optimizations."""
    # Import performance optimizations
    from app.services.response_cache import (
        get_cached_response, should_use_fast_model
    )

    # Prepare messages early for cache lookup (need for context hash)
    sys_prompt_for_cache = req.system_prompt
    if (req.locale or "en").lower().startswith("ja"):
        sys_prompt_for_cache = (sys_prompt_for_cache or "").rstrip() + _JAPANESE_DIRECTIVE
    
    messages_for_cache, _ = await prepare_conversation_messages_with_memory(
        req.prompt, sys_prompt_for_cache, req.conversation_id or "", False, str(user.id)
    )

    # Check cache first for performance boost (only for non-reset, existing conversations)
    cached_response = None
    if not req.reset and not req.conversation_id:
        cached_response = get_cached_response(req.prompt, req.deployment or DEFAULT_MODEL, messages_for_cache)
    if cached_response:
        # Return cached response as stream
        async def serve_cached():
            yield f"data: {json.dumps({'type': 'start', 'conversation_id': cached_response.get('conversation_id', ''), 'model': 'cached', 'cached': True})}\n\n"
            content = cached_response.get('content', '')
            # Stream cached content in chunks for consistent UX
            chunk_size = 50
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                yield f"data: {json.dumps({'type': 'content', 'delta': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(serve_cached(), media_type="text/event-stream")

    # Reuse messages from cache lookup, or re-fetch if reset is requested
    if req.reset:
        messages, conv_id = await prepare_conversation_messages_with_memory(
            req.prompt, sys_prompt_for_cache, req.conversation_id or "", req.reset, str(user.id)
        )
    else:
        # Already fetched for cache lookup
        messages = messages_for_cache
        # Generate new conversation ID
        conv_id = str(uuid.uuid4())

    # Performance optimization: detect simple queries and use fast model
    is_simple, query_type = is_simple_query(req.prompt)
    model_key = req.deployment or DEFAULT_MODEL
    
    if is_simple and not req.deployment:
        fast_model = get_fast_model_recommendation(req.prompt)
        if fast_model:
            logger.info(f"Using fast model {fast_model} for simple query type: {query_type}")
            model_key = fast_model
    
    # Performance optimization: skip heavy tools for simple queries
    skip_heavy = should_skip_rag_and_web_search(req.prompt)
    
    # Use ML tool selection if enabled, otherwise fall back to rule-based
    if ML_TOOL_SELECTION_ENABLED:
        selected_tools, tool_metadata = build_tools_for_request_ml(
            req.prompt, 
            getattr(req, "capabilities", None), 
            skip_heavy_tools=skip_heavy,
            use_ml=True,
            fallback_to_rules=True
        )
        logger.info(
            f"Tool selection: method={tool_metadata.get('method')}, "
            f"confidence={tool_metadata.get('confidence')}, "
            f"count={tool_metadata.get('tools_count')}"
        )
    else:
        selected_tools = build_tools_for_request(req.prompt, getattr(req, "capabilities", None), skip_heavy_tools=skip_heavy)
        tool_metadata = {'method': 'rule-based', 'ml_enabled': False}
    
    selected_tool_names = {tool.get("function", {}).get("name") for tool in selected_tools}
    logger.debug(
        "Selected tools for request %s: %s",
        conv_id,
        sorted(name for name in selected_tool_names if name),
    )

    # Validate model exists before proceeding
    from app.core.config import AVAILABLE_MODELS
    if model_key not in AVAILABLE_MODELS:
        logger.error(f"Invalid model requested: {model_key}")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model '{model_key}'. Available models: {', '.join(AVAILABLE_MODELS.keys())}"
        )

    # Use the enhanced model selection system with optimizations
    from app.services.openai_client import get_client_for_model

    # Set timeout based on query complexity
    timeout = 30 if is_simple else None

    try:
        client, model_name, resolved_config = get_client_for_model(model_key, timeout)
        logger.info(
            f"Using model: {model_key} -> {model_name} (timeout: {resolved_config.get('timeout', 'default')})"
        )
    except Exception as e:
        logger.error(f"Failed to get client for model {model_key}: {e}")
        raise HTTPException(status_code=500, detail=f"Model '{model_key}' is not available: {e}")

    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        nonlocal messages
        tool_call_results: List[Dict[str, Any]] = []
        suggested_tool_rounds = 0
        full_content = ""

        logger.debug("[chat_stream] Entered generate_stream for conversation %s model_key=%s", conv_id, model_key)

        # Track which tools we've already announced as running to the client
        announced_tools: set[str] = set()

        from app.core.config import AVAILABLE_MODELS
        model_metadata = AVAILABLE_MODELS.get(model_key, {})

        token_param_key = model_metadata.get("completion_token_param")
        if not token_param_key:
            model_identifier = model_metadata.get("model") or model_name
            if isinstance(model_identifier, str) and (model_identifier.startswith("gpt-5") or model_identifier.startswith("gpt-o3")):
                token_param_key = "max_completion_tokens"
            else:
                token_param_key = "max_tokens"

        def _apply_token_limit(params: Dict[str, Any], value: int) -> None:
            """Apply the correct token limit parameter based on model requirements."""
            params.pop("max_tokens", None)
            params.pop("max_completion_tokens", None)
            params[token_param_key] = value

        def _resolve_timeout(default: int = 60) -> int:
            return resolved_config.get("timeout", model_metadata.get("timeout", default))

        def _resolve_temperature(default: float = 0.7) -> float:
            if model_metadata.get("temperature_fixed"):
                return model_metadata["temperature"]
            return resolved_config.get("temperature", model_metadata.get("temperature", default))

        def _resolve_max_tokens(default: int) -> int:
            if "max_completion_tokens" in resolved_config:
                return resolved_config["max_completion_tokens"]
            if "max_completion_tokens" in model_metadata:
                return model_metadata["max_completion_tokens"]
            if "max_tokens" in model_metadata:
                return model_metadata["max_tokens"]
            return default

        def _resolve_max_tokens_with_cap(default: int) -> int:
            resolved = _resolve_max_tokens(default)
            return min(resolved, default)

        def _execute_single_tool(tc: Dict[str, Any]) -> tuple[str, Dict[str, Any], Optional[str]]:
            """Execute a single tool call. Returns (tool_call_id, result, error_message)."""
            try:
                tool_call_id = tc.get("id")
                fn = tc.get("function") or {}
                name = fn.get("name")
                raw_args = fn.get("arguments") or "{}"

                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except Exception as e:
                    logger.warning(f"Failed to parse tool arguments for {name}: {e}")
                    args = {}

                impl = TOOL_REGISTRY.get(name)
                if not impl:
                    result = {"error": f"unknown tool: {name}"}
                else:
                    start_time = time.perf_counter()
                    result = impl(**(args or {}))
                    execution_time = time.perf_counter() - start_time
                    
                    # Special handling: summarize augmented news to keep context small 
                    if name == "get_augmented_news" and isinstance(result, dict):
                        summary = build_news_summary(result)
                        result = {**summary}
                    # Truncate large generic tool results to control costs
                    elif isinstance(result, dict) and "items" in result and name != "get_augmented_news":
                        if len(result.get("items", [])) > 5:
                            result["items"] = result["items"][:5]
                            result["truncated"] = True
                    
                    logger.debug(f"Tool {name} executed in {execution_time:.3f}s")

                # Apply perplexity_search specific sanitation
                if name == "perplexity_search" and isinstance(result, dict):
                    result = sanitize_perplexity_result(result)

                return tool_call_id, result, None
            except Exception as e:
                logger.error(f"Tool execution error for {tc}: {e}")
                return tc.get("id"), {"error": str(e)}, str(e)

        async def _run_tools_async(tc_list: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
            """Execute tool calls asynchronously with parallelization and yield progress updates."""
            nonlocal messages, tool_call_results
            
            # Validate tool calls first
            valid_tool_calls = []
            for tc in tc_list:
                try:
                    if (tc.get("type") or "function") != "function":
                        continue
                    if not tc.get("id") or not tc.get("function", {}).get("name"):
                        logger.warning(f"Skipping invalid tool call: {tc}")
                        continue
                    valid_tool_calls.append(tc)
                except Exception as e:
                    logger.error(f"Error validating tool call {tc}: {e}")
                    continue
            
            if not valid_tool_calls:
                return
            
            # Submit all tool calls to thread pool for parallel execution
            loop = asyncio.get_running_loop()
            
            # Create tasks with metadata
            task_to_tc = {}
            for tc in valid_tool_calls:
                task = loop.run_in_executor(_tool_executor, _execute_single_tool, tc)
                task_to_tc[task] = tc
            
            # Process results as they complete using asyncio.wait with FIRST_COMPLETED
            completed_tools = []
            pending_tasks = set(task_to_tc.keys())
            
            while pending_tasks:
                done, pending_tasks = await asyncio.wait(
                    pending_tasks, 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    try:
                        tool_call_id, result, error = await task
                        tc = task_to_tc[task]
                        name = tc.get("function", {}).get("name", "unknown")
                        
                        # Truncate tool result content for token management, preserving web search citations
                        if name in _WEB_SEARCH_TOOL_NAMES and isinstance(result, dict):
                            payload = build_web_search_tool_payload(result)
                            result_str = safe_tool_result_json(payload)
                            if estimate_tokens(result_str) > 768:
                                compact_payload = dict(payload)
                                compact_payload.pop("top_sources", None)
                                result_str = safe_tool_result_json(compact_payload)
                        else:
                            result_str = safe_tool_result_json(result)
                            if estimate_tokens(result_str) > 512:
                                result_str = result_str[:2000] + "... [truncated]"

                        # Add tool result message
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": result_str,
                        })
                        
                        # Track results and yield updates
                        if error:
                            tool_call_results.append({"id": tool_call_id, "name": name, "error": error})
                            yield f"data: {_PRECOMPILED_RESPONSES['tool_error'](name, error)}\n\n"
                        else:
                            tool_call_results.append({"id": tool_call_id, "name": name, "result": result})
                            yield f"data: {_PRECOMPILED_RESPONSES['tool_completed'](name)}\n\n"
                        
                        completed_tools.append(name)
                        
                    except Exception as e:
                        logger.error(f"Error processing completed tool task: {e}")
                        tc = task_to_tc.get(task, {})
                        name = tc.get("function", {}).get("name", "unknown")
                        yield f"data: {_PRECOMPILED_RESPONSES['tool_error'](name, str(e))}\n\n"
            
            logger.info(f"Completed {len(completed_tools)} tool calls: {completed_tools}")

        # Send initial metadata with pre-compiled response
        yield f"data: {_PRECOMPILED_RESPONSES['start'](conv_id, model_name)}\n\n"

        try:
            # Handle tool calling with streaming (1 round only for faster responses)
            for iteration in range(1):
                # Check token budget before each API call - use cached token calculations
                from app.utils.token_utils import calculate_total_tokens, optimize_messages_for_token_budget
                
                current_tokens = calculate_total_tokens(messages)
                if current_tokens > MAX_TOKENS_PER_TURN * 0.95:  # Increased from 0.9 to 0.95
                    logger.warning(f"Approaching token limit ({current_tokens}), optimizing messages")
                    messages = optimize_messages_for_token_budget(
                        messages, 
                        int(MAX_TOKENS_PER_TURN * 0.9),
                        preserve_system=True,
                        preserve_recent_user=2  # Keep last 2 user messages
                    )
                
                # Final validation: ensure complete tool call/response pairs before API call
                from app.utils.conversation import _validate_message_sequence
                messages = _validate_message_sequence(messages)

                # Create streaming completion
                try:
                    model_timeout = _resolve_timeout()
                    # Increase max_tokens for better AI responses, especially after tool execution
                    # Normalize config key: some configs may still name it max_completion_tokens
                    max_tokens = _resolve_max_tokens(min(4000, MAX_TOKENS_PER_TURN // 2))
                    temperature = _resolve_temperature()

                    logger.info(
                        f"Creating stream for {model_key} with timeout={model_timeout}, "
                        f"{token_param_key}={max_tokens}"
                    )

                    # Build completion parameters
                    completion_params = {
                        "model": model_name,
                        "messages": messages,
                        "tools": selected_tools,
                        "tool_choice": "auto",
                        "stream": True,
                        "timeout": model_timeout,
                    }
                    _apply_token_limit(completion_params, max_tokens)
                    if not isinstance(client, AsyncAzureOpenAI):
                        # Optimize OpenAI streaming performance; Azure does not support stream_options
                        completion_params["stream_options"] = {"include_usage": False}
                    
                    # Only include temperature if model supports it
                    if not model_metadata.get("temperature_fixed"):
                        completion_params["temperature"] = temperature
                    
                    stream = await client.chat.completions.create(**completion_params)
                except Exception as api_error:
                    logger.error(f"Failed to create streaming completion with {model_name}: {api_error}")
                    # Fallback to non-streaming once
                    try:
                        fallback_default = min(2000, MAX_TOKENS_PER_TURN // 4)
                        fallback_params = {
                            "model": model_name,
                            "messages": messages,
                            "tools": selected_tools,
                            "tool_choice": "auto",
                            "timeout": _resolve_timeout(),
                        }
                        _apply_token_limit(
                            fallback_params,
                            _resolve_max_tokens_with_cap(fallback_default)
                        )
                        # Only include temperature if model supports it
                        if not model_metadata.get("temperature_fixed"):
                            fallback_params["temperature"] = _resolve_temperature()
                        
                        completion = await client.chat.completions.create(**fallback_params)
                        choice = (completion.choices or [None])[0]
                        msg = getattr(choice, "message", None)
                        content = getattr(msg, "content", "") if msg else ""
                        # Normalize content if it's a structured payload
                        if isinstance(content, list):
                            content = "".join(normalize_content_piece(p) for p in content)
                        content = content or ""
                        if content:
                            full_content += content
                            yield f"data: {json.dumps({'type': 'content', 'delta': content})}\n\n"
                            messages.append({"role": "assistant", "content": content})
                            break
                        else:
                            yield f"data: {json.dumps({'type': 'error', 'error': 'No content returned by model'})}\n\n"
                            return
                    except Exception as e2:
                        yield f"data: {json.dumps({'type': 'error', 'error': f'Model API error: {str(e2)}'})}\n\n"
                        return

                assistant_msg: Dict[str, Any] = {"role": "assistant", "content": ""}
                collected_tool_calls = []

                # Process streaming chunks with timeout protection
                try:
                    chunk_count = 0
                    any_text = False
                    for chunk in stream:
                        chunk_count += 1

                        # Safety check for runaway streams
                        if chunk_count > 1000:
                            logger.warning(f"Stream exceeded chunk limit for model {model_name}")
                            break

                        if not chunk.choices:
                            continue

                        delta = chunk.choices[0].delta

                        # Handle content streaming (support string or list shards)
                        if hasattr(delta, 'content') and delta.content:
                            if isinstance(delta.content, list):
                                # Join any textual pieces
                                piece_text = "".join(normalize_content_piece(p) for p in delta.content)
                                if piece_text:
                                    # Skip raw tool call JSON outputs from GPT-5
                                    if is_raw_tool_call_output(piece_text):
                                        logger.debug(f"Filtered out raw tool call JSON from {model_name}")
                                        continue
                                    
                                    # Convert LaTeX format for frontend compatibility
                                    converted_text = _convert_latex_format(piece_text)
                                    assistant_msg["content"] += converted_text
                                    full_content += converted_text
                                    any_text = True
                                    # Use pre-compiled response for better performance
                                    yield f"data: {_PRECOMPILED_RESPONSES['content'](converted_text)}\n\n"
                            elif isinstance(delta.content, str):
                                # Skip raw tool call JSON outputs from GPT-5
                                if is_raw_tool_call_output(delta.content):
                                    logger.debug(f"Filtered out raw tool call JSON from {model_name}")
                                    continue
                                
                                # Convert LaTeX format for frontend compatibility
                                converted_text = _convert_latex_format(delta.content)
                                assistant_msg["content"] += converted_text
                                full_content += converted_text
                                any_text = True
                                # Use pre-compiled response for better performance
                                yield f"data: {_PRECOMPILED_RESPONSES['content'](converted_text)}\n\n"

                        # Handle tool calls streamed by the model
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            newly_seen: List[str] = []
                            for tc in delta.tool_calls:
                                # Extend or create tool call entries
                                idx = 0
                                try:
                                    idx = int(getattr(tc, 'index', 0) or 0)
                                except Exception:
                                    idx = 0
                                while len(collected_tool_calls) <= idx:
                                    collected_tool_calls.append({
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })

                                if getattr(tc, 'id', None):
                                    collected_tool_calls[idx]["id"] = tc.id
                                if getattr(tc, 'function', None) and getattr(tc.function, 'name', None):
                                    name = tc.function.name
                                    collected_tool_calls[idx]["function"]["name"] = name
                                    if name and name not in announced_tools:
                                        announced_tools.add(name)
                                        newly_seen.append(name)
                                        # Emit an immediate running status for this tool with pre-compiled response
                                        yield f"data: {_PRECOMPILED_RESPONSES['tool_running'](name)}\n\n"
                                if getattr(tc, 'function', None) and getattr(tc.function, 'arguments', None):
                                    collected_tool_calls[idx]["function"]["arguments"] += tc.function.arguments

                            # Also emit a batched tool list once, when we first see any
                            if newly_seen:
                                yield f"data: {json.dumps({'type': 'tool_calls', 'tools': newly_seen})}\n\n"
                except Exception as stream_error:
                    logger.error(f"Error processing stream chunks for model {model_name}: {stream_error}")
                    yield f"data: {json.dumps({'type': 'error', 'error': f'Streaming error: {str(stream_error)}'})}\n\n"
                    return

                # If we have tool calls, execute them
                if collected_tool_calls:
                    # Validate all tool calls have required fields before adding to messages
                    valid_tool_calls = []
                    for tc in collected_tool_calls:
                        if tc.get("id") and tc.get("function", {}).get("name"):
                            valid_tool_calls.append(tc)
                        else:
                            logger.warning(f"Skipping invalid tool call: {tc}")
                    
                    if valid_tool_calls:
                        assistant_msg["tool_calls"] = valid_tool_calls
                        messages.append(assistant_msg)

                        # If not yet announced (e.g., non-stream tool_calls), notify
                        yield f"data: {json.dumps({'type': 'tool_calls', 'tools': [tc['function']['name'] for tc in valid_tool_calls]})}\n\n"

                        # Execute tools asynchronously and yield updates as they complete
                        async for update in _run_tools_async(valid_tool_calls):
                            yield update

                        # Continue loop for next iteration
                        continue
                    else:
                        logger.warning("All tool calls were invalid, treating as regular assistant message")
                        # Treat as regular message if no valid tool calls
                        messages.append(assistant_msg)
                        break
                else:
                    # Pseudo tool call fallback: parse assistant content for OSS-style tool markup
                    if assistant_msg.get("content"):
                        pseudo_calls = extract_pseudo_tool_calls(assistant_msg["content"]) or []
                        if pseudo_calls:
                            # Strip markup from content for display
                            assistant_msg["content"] = strip_pseudo_tool_markup(assistant_msg["content"]) or ""
                            messages.append(assistant_msg)
                            yield f"data: {json.dumps({'type': 'content', 'delta': assistant_msg['content']})}\n\n"

                            # Notify and execute pseudo tool calls
                            yield f"data: {json.dumps({'type': 'tool_calls', 'tools': [tc['function']['name'] for tc in pseudo_calls], 'pseudo': True})}\n\n"
                            async for update in _run_tools_async(pseudo_calls):
                                yield update
                            # Continue loop to allow the model to use tool results
                            continue

                        suggested_calls = extract_suggested_tool_calls(assistant_msg["content"]) or []
                        if suggested_calls:
                            if suggested_tool_rounds >= MAX_SUGGESTED_TOOL_ROUNDS:
                                logger.info(
                                    "Skipping suggested tool calls due to round limit (round=%s, tools=%s)",
                                    suggested_tool_rounds,
                                    [call["function"]["name"] for call in suggested_calls],
                                )
                            else:
                                next_round = suggested_tool_rounds + 1
                                augmented_calls: List[Dict[str, Any]] = []
                                for idx, call in enumerate(suggested_calls, start=1):
                                    function_payload = dict(call.get("function", {}))
                                    augmented_calls.append({
                                        "id": f"suggested-{next_round}-{idx}",
                                        "type": "function",
                                        "function": function_payload,
                                    })

                                assistant_msg["tool_calls"] = augmented_calls
                                messages.append(assistant_msg)
                                suggested_tool_rounds = next_round

                                yield f"data: {json.dumps({'type': 'tool_calls', 'tools': [tc['function']['name'] for tc in augmented_calls], 'suggested': True, 'round': suggested_tool_rounds})}\n\n"
                                async for update in _run_tools_async(augmented_calls):
                                    yield update
                                continue

                    # No tool calls; if no streamed text, fallback to single-shot completion
                    if not assistant_msg.get("content"):
                        try:
                            no_stream_default = min(2000, MAX_TOKENS_PER_TURN // 4)
                            no_stream_params = {
                                "model": model_name,
                                "messages": messages,
                                "tools": selected_tools,
                                "tool_choice": "auto",
                                "timeout": _resolve_timeout(),
                            }
                            _apply_token_limit(
                                no_stream_params,
                                _resolve_max_tokens_with_cap(no_stream_default)
                            )
                            if not model_metadata.get("temperature_fixed"):
                                no_stream_params["temperature"] = _resolve_temperature()
                                
                            completion = await client.chat.completions.create(**no_stream_params)
                            choice = (completion.choices or [None])[0]
                            msg = getattr(choice, "message", None)
                            content = getattr(msg, "content", "") if msg else ""
                            if isinstance(content, list):
                                content = "".join(normalize_content_piece(p) for p in content)
                            content = content or ""
                            if content:
                                converted_content = _convert_latex_format(content)
                                assistant_msg["content"] = converted_content
                                full_content += converted_content
                                yield f"data: {json.dumps({'type': 'content', 'delta': converted_content})}\n\n"
                            else:
                                logger.info("Model returned empty content after stream; sending empty message")
                        except Exception as e:
                            logger.error(f"Fallback non-stream completion failed: {e}")

                    messages.append(assistant_msg)
                    break

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            return

        # FORCE AI to generate response when tools were executed - don't just use fallback
        if tool_call_results and not full_content:
            # Try one more time to get AI response with emphasis on responding
            try:
                # Add a message to encourage AI response
                response_prompt = "Based on the tool results above, please provide a comprehensive response to the user's question in the appropriate language."
                messages.append({"role": "user", "content": response_prompt})
                
                retry_default = min(4000, MAX_TOKENS_PER_TURN // 2)
                ai_retry_params = {
                    "model": model_name,
                    "messages": messages,
                    "timeout": _resolve_timeout(),
                }
                _apply_token_limit(
                    ai_retry_params,
                    _resolve_max_tokens_with_cap(retry_default)
                )
                if not model_metadata.get("temperature_fixed"):
                    ai_retry_params["temperature"] = _resolve_temperature()
                    
                completion = await client.chat.completions.create(**ai_retry_params)
                choice = (completion.choices or [None])[0]
                msg = getattr(choice, "message", None)
                content = getattr(msg, "content", "") if msg else ""
                if isinstance(content, list):
                    content = "".join(normalize_content_piece(p) for p in content)
                
                if content and content.strip():
                    converted_content = _convert_latex_format(content)
                    full_content += converted_content
                    yield f"data: {json.dumps({'type': 'content', 'delta': converted_content})}\n\n"
                    messages.append({"role": "assistant", "content": converted_content})
                    
            except Exception as ai_retry_error:
                logger.error(f"Failed to force AI response: {ai_retry_error}")
                
        # If AI still didn't respond after retry, use fallback
        if tool_call_results and not full_content:
            # Fallback: provide a comprehensive summary when AI completely fails
            try:
                # Check if this is a Japanese request to format appropriately
                is_japanese = False
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if any(ord(c) > 0x3000 for c in content):  # Contains Japanese characters
                            is_japanese = True
                            break
                
                fallback_parts = []
                
                # Process all tool results, not just the last one
                for tool_result in tool_call_results:
                    if not isinstance(tool_result, dict):
                        continue
                    
                    name = tool_result.get("name")
                    res = tool_result.get("result") or {}
                    
                    if name == "get_company_profile" and isinstance(res, dict):
                        if is_japanese:
                            fallback_parts.append(human_preview_company_profile_jp(res))
                        else:
                            fallback_parts.append(human_preview_company_profile(res))
                    elif name == "get_stock_quote" and isinstance(res, dict):
                        if is_japanese:
                            fallback_parts.append(human_preview_quote_jp(res))
                        else:
                            fallback_parts.append(human_preview_quote(res))
                    elif name == "get_historical_prices" and isinstance(res, dict):
                        if is_japanese:
                            fallback_parts.append(human_preview_historical_jp(res))
                        else:
                            fallback_parts.append(human_preview_historical(res))
                
                # Combine all parts or use generic fallback
                if fallback_parts:
                    fallback = "\n\n".join(fallback_parts)
                else:
                    # Generic JSON fallback for unhandled tools
                    try:
                        last = tool_call_results[-1] if tool_call_results else None
                        fallback = safe_tool_result_json(last.get("result"))[:400] + "..."  # soft limit
                    except Exception:
                        fallback = "(tool result received)"
                
                converted_fallback = _convert_latex_format(fallback)
                full_content += converted_fallback
                yield f"data: {json.dumps({'type': 'content', 'delta': converted_fallback})}\n\n"
                # Also append to messages so it persists in history
                messages.append({"role": "assistant", "content": converted_fallback})
            except Exception as e:
                logger.error(f"Fallback generation error: {e}")
                pass

        # Store conversation with enhanced memory
        try:
            await store_conversation_messages_with_memory(conv_id, messages, str(user.id))

            db.add(Log(
                user_id=int(user.id),
                action="chat_stream",
                conversation_id=conv_id,
                prompt=req.prompt[:1000] if len(req.prompt) > 1000 else req.prompt,
                response=full_content[:2000] if len(full_content) > 2000 else full_content,
                tool_calls=tool_call_results
            ))
            db.commit()
        except Exception as e:
            logger.error(f"Database error in streaming: {e}")
            db.rollback()

        # Cache the response if it's a new conversation (same logic as non-streaming endpoint)
        if not req.reset and not req.conversation_id and full_content:
            try:
                from app.services.response_cache import cache_response
                cache_data = {
                    'content': full_content,
                    'tool_calls': tool_call_results if tool_call_results else None,
                    'conversation_id': conv_id
                }
                cache_response(req.prompt, model_key, cache_data, messages)
                logger.debug(f"Cached streaming response for conversation {conv_id}")
            except Exception as e:
                logger.warning(f"Failed to cache streaming response: {e}")

        # Send completion status
        completion_data = {
            'type': 'done', 
            'conversation_id': conv_id, 
            'tool_calls': tool_call_results
        }
        yield f"data: {json.dumps(completion_data)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """Main chat endpoint with AI conversation, tool calling, and enhanced memory."""
    # Track execution time for ML logging
    start_time = time.time()
    
    # If streaming is requested, redirect to streaming endpoint
    if req.stream:
        raise HTTPException(status_code=400, detail="Use /chat/stream endpoint for streaming responses")

    # Use the new enhanced model selection system
    from app.services.openai_client import get_client_for_model

    # Resolve model key - use deployment parameter as model key, fallback to default
    model_key = req.deployment or DEFAULT_MODEL
    
    # Performance optimization: use fast model for simple queries
    is_simple, query_type = is_simple_query(req.prompt)
    if is_simple and not req.deployment:
        # User didn't specify a model, so we can use fast model
        fast_model = get_fast_model_recommendation(req.prompt)
        if fast_model:
            logger.info(f"Using fast model {fast_model} for simple query type: {query_type}")
            model_key = fast_model
    
    # Check request cache for identical queries (performance optimization)
    sys_prompt = req.system_prompt or ""
    if not req.reset and not req.conversation_id:
        # Only cache for new conversations without reset
        cached = get_cached_response(req.prompt, model_key, sys_prompt)
        if cached:
            logger.info(f"Returning cached response for prompt hash")
            return ChatResponse(**cached)
        
        # Check if an identical request is already being processed
        if is_request_in_flight(req.prompt, model_key, sys_prompt):
            logger.info(f"Identical request already in flight, waiting...")
            # Wait a bit and check cache again (the in-flight request might complete)
            import asyncio
            await asyncio.sleep(0.5)
            cached = get_cached_response(req.prompt, model_key, sys_prompt)
            if cached:
                return ChatResponse(**cached)
        
        # Mark this request as in-flight
        mark_request_in_flight(req.prompt, model_key, sys_prompt)

    # Use enhanced memory-aware message preparation
    # Inject locale-specific instruction into system prompt if requested
    sys_prompt = req.system_prompt
    if (req.locale or "en").lower().startswith("ja"):
        sys_prompt = (sys_prompt or "").rstrip() + _JAPANESE_DIRECTIVE

    messages, conv_id = await prepare_conversation_messages_with_memory(
        req.prompt, sys_prompt, req.conversation_id or "", req.reset, str(user.id), model_key
    )

    # Performance optimization: skip heavy tools for simple queries
    skip_heavy = should_skip_rag_and_web_search(req.prompt)
    
    # Use ML tool selection if enabled, otherwise fall back to rule-based
    if ML_TOOL_SELECTION_ENABLED:
        selected_tools, tool_metadata = build_tools_for_request_ml(
            req.prompt, 
            getattr(req, "capabilities", None), 
            skip_heavy_tools=skip_heavy,
            use_ml=True,
            fallback_to_rules=True
        )
        logger.info(
            f"Tool selection: method={tool_metadata.get('method')}, "
            f"confidence={tool_metadata.get('confidence')}, "
            f"count={tool_metadata.get('tools_count')}"
        )
    else:
        selected_tools = build_tools_for_request(req.prompt, getattr(req, "capabilities", None), skip_heavy_tools=skip_heavy)
        tool_metadata = {'method': 'rule-based', 'ml_enabled': False}
    
    selected_tool_names = {tool.get("function", {}).get("name") for tool in selected_tools}
    logger.debug(
        "Selected tools for request %s: %s",
        conv_id,
        sorted(name for name in selected_tool_names if name),
    )

    try:
        client, model_name, client_config = get_client_for_model(model_key)
        logger.info(f"Using model: {model_key} -> {model_name}")
    except Exception as e:
        logger.error(f"Failed to get client for model {model_key}: {e}")
        raise HTTPException(status_code=500, detail=f"Model '{model_key}' is not available: {e}")

    tool_call_results: List[Dict[str, Any]] = []
    suggested_tool_rounds = 0

    from app.core.config import AVAILABLE_MODELS
    model_metadata = AVAILABLE_MODELS.get(model_key, {})

    token_param_key = model_metadata.get("completion_token_param")
    if not token_param_key:
        model_identifier = model_metadata.get("model") or model_name
        if isinstance(model_identifier, str) and model_identifier.startswith("gpt-5"):
            token_param_key = "max_completion_tokens"
        else:
            token_param_key = "max_tokens"

    def apply_token_limit(params: Dict[str, Any], value: int) -> None:
        params.pop("max_tokens", None)
        params.pop("max_completion_tokens", None)
        params[token_param_key] = value

    def resolve_timeout(default: int = 60) -> int:
        return client_config.get("timeout", model_metadata.get("timeout", default))

    def resolve_temperature(default: float = 0.7) -> float:
        if model_metadata.get("temperature_fixed"):
            return model_metadata["temperature"]
        return client_config.get("temperature", model_metadata.get("temperature", default))

    def resolve_max_tokens(default: int) -> int:
        if "max_completion_tokens" in client_config:
            return client_config["max_completion_tokens"]
        if "max_completion_tokens" in model_metadata:
            return model_metadata["max_completion_tokens"]
        if "max_tokens" in model_metadata:
            return model_metadata["max_tokens"]
        return default

    def resolve_max_tokens_with_cap(default: int) -> int:
        resolved = resolve_max_tokens(default)
        return min(resolved, default)


    def _run_tools(tc_list: List[Dict[str, Any]], max_rounds: int = 2):
        """Execute tool calls and append results to messages."""
        pending_calls: List[Dict[str, Any]] = list(tc_list or [])
        rounds_executed = 0

        while pending_calls and rounds_executed < max_rounds:
            rounds_executed += 1

            # Only process valid tool calls to avoid OpenAI validation errors
            valid_tool_calls: List[Dict[str, Any]] = []
            for tc in pending_calls:
                try:
                    if (tc.get("type") or "function") != "function":
                        continue

                    tool_call_id = tc.get("id")
                    if not tool_call_id:
                        logger.warning(f"Tool call missing ID: {tc}")
                        continue

                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    if not name:
                        logger.warning(f"Tool call missing function name: {tc}")
                        continue

                    valid_tool_calls.append(tc)
                except Exception as e:
                    logger.error(f"Error validating tool call {tc}: {e}")
                    continue

            if not valid_tool_calls:
                break

            pending_calls = []  # Reset; may be repopulated in future enhancements

            for tc in valid_tool_calls:
                try:
                    tool_call_id = tc.get("id")
                    fn = tc.get("function") or {}
                    name = fn.get("name")

                    raw_args = fn.get("arguments") or "{}"

                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception as e:
                        logger.warning(f"Failed to parse tool arguments for {name}: {e}")
                        args = {}

                    impl = TOOL_REGISTRY.get(name)
                    if not impl:
                        result = {"error": f"unknown tool: {name}"}
                    else:
                        try:
                            result = impl(**(args or {}))
                            if name == "get_augmented_news" and isinstance(result, dict):
                                result = _build_news_summary(result)
                            if isinstance(result, dict) and "items" in result and name != "get_augmented_news":
                                if len(result.get("items", [])) > 5:
                                    result["items"] = result["items"][:5]
                                    result["truncated"] = True
                            if name == "perplexity_search" and isinstance(result, dict):
                                result = _sanitize_perplexity_result(result)
                        except Exception as e:
                            logger.error(f"Tool execution error for {name}: {e}")
                            result = {"error": str(e)}

                    if name in _WEB_SEARCH_TOOL_NAMES and isinstance(result, dict):
                        payload = _build_web_search_tool_payload(result)
                        result_str = _safe_tool_result_json(payload)
                        if estimate_tokens(result_str) > 768:
                            compact_payload = dict(payload)
                            compact_payload.pop("top_sources", None)
                            result_str = _safe_tool_result_json(compact_payload)
                    else:
                        result_str = _safe_tool_result_json(result)
                        if estimate_tokens(result_str) > 512:
                            result_str = result_str[:2000] + "... [truncated]"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_str,
                    })
                    tool_call_results.append({"id": tool_call_id, "name": name, "result": result})
                except Exception as e:
                    logger.error(f"Unexpected error processing tool call {tc}: {e}")
                    tool_call_id = tc.get("id") if isinstance(tc, dict) else None
                    if tool_call_id:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps({"error": f"Tool execution failed: {str(e)}"}),
                        })
                    tool_call_results.append({
                        "id": tool_call_id,
                        "name": (tc.get("function") or {}).get("name") if isinstance(tc, dict) else "unknown",
                        "error": str(e)
                    })

    # Single round of tool calling for faster responses
    final_content: Optional[str] = None
    for iteration in range(1):
        try:
            # Check token budget before each API call - use cached token calculations
            from app.utils.token_utils import calculate_total_tokens, optimize_messages_for_token_budget
            
            current_tokens = calculate_total_tokens(messages)
            if current_tokens > MAX_TOKENS_PER_TURN * 0.95:  # Increased from 0.9 to 0.95
                logger.warning(f"Approaching token limit ({current_tokens}), optimizing messages")
                messages = optimize_messages_for_token_budget(
                    messages, 
                    int(MAX_TOKENS_PER_TURN * 0.9),
                    preserve_system=True,
                    preserve_recent_user=2  # Keep last 2 user messages
                )
            
            # Final validation: ensure complete tool call/response pairs before API call
            from app.utils.conversation import _validate_message_sequence
            messages = _validate_message_sequence(messages)

            completion_params = {
                "model": model_name,
                "messages": messages,
                "tools": selected_tools,
                "tool_choice": "auto",
                "timeout": resolve_timeout(),
            }
            apply_token_limit(
                completion_params,
                resolve_max_tokens(min(4000, MAX_TOKENS_PER_TURN // 2))
            )
            
            # Only include temperature if model supports it
            if not model_metadata.get("temperature_fixed"):
                completion_params["temperature"] = resolve_temperature()
                
            completion = await client.chat.completions.create(**completion_params)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

        choice = (completion.choices or [None])[0]
        if not choice or not getattr(choice, "message", None):
            raise HTTPException(status_code=500, detail="No completion returned")

        msg = getattr(choice, "message", None)
        # Normalize to dict
        raw_content = getattr(msg, "content", None) if msg else ""
        if isinstance(raw_content, list):
            # Join structured content parts if any
            content_joined = "".join(
                (p.get("text") if isinstance(p, dict) and isinstance(p.get("text"), str) else str(p))
                for p in raw_content
            )
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content_joined}
        else:
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": raw_content}

        # Handle tool calls
        tc = []
        try:
            tc = [
                {
                    "id": t.id,
                    "type": t.type,
                    "function": {"name": t.function.name, "arguments": t.function.arguments},
                }
                for t in (msg.tool_calls or [])
            ] if msg else []
        except Exception:
            # Already dicts
            tc = getattr(msg, "tool_calls", None) or []

        if tc:
            # Validate all tool calls have required fields before adding to messages
            valid_tool_calls = []
            for tool_call in tc:
                if tool_call.get("id") and tool_call.get("function", {}).get("name"):
                    valid_tool_calls.append(tool_call)
                else:
                    logger.warning(f"Skipping invalid tool call: {tool_call}")
            
            if valid_tool_calls:
                assistant_msg["tool_calls"] = valid_tool_calls
                messages.append(assistant_msg)
                _run_tools(valid_tool_calls)
                # Continue the loop to send tool outputs back to the model
                continue
            else:
                logger.warning("All tool calls were invalid, treating as regular assistant message")
                # Treat as regular message if no valid tool calls
                messages.append(assistant_msg)

        # Fallback: check for pseudo tool-calls embedded in text
        if assistant_msg.get("content"):
            pseudo_calls = _extract_pseudo_tool_calls(assistant_msg["content"]) or []
            if pseudo_calls:
                assistant_msg["content"] = _strip_pseudo_tool_markup(assistant_msg["content"]) or ""
                messages.append(assistant_msg)
                _run_tools(pseudo_calls)
                # Continue loop to send tool outputs back to the model
                continue

        if assistant_msg.get("content"):
            suggested_calls = _extract_suggested_tool_calls(assistant_msg["content"]) or []
            if suggested_calls:
                if suggested_tool_rounds >= MAX_SUGGESTED_TOOL_ROUNDS:
                    logger.info(
                        "Skipping suggested tool calls in sync flow due to round limit (round=%s, tools=%s)",
                        suggested_tool_rounds,
                        [call["function"]["name"] for call in suggested_calls],
                    )
                else:
                    next_round = suggested_tool_rounds + 1
                    augmented_calls: List[Dict[str, Any]] = []
                    for idx, call in enumerate(suggested_calls, start=1):
                        function_payload = dict(call.get("function", {}))
                        augmented_calls.append({
                            "id": f"suggested-{next_round}-{idx}",
                            "type": "function",
                            "function": function_payload,
                        })

                    assistant_msg["tool_calls"] = augmented_calls
                    messages.append(assistant_msg)
                    suggested_tool_rounds = next_round
                    _run_tools(augmented_calls)
                    continue

        # No tool calls; finalize
        final_content = assistant_msg.get("content") or ""
        if isinstance(final_content, list):
            final_content = "".join(
                (p.get("text") if isinstance(p, dict) and isinstance(p.get("text"), str) else str(p))
                for p in final_content
            )
        
        # Filter out raw tool call JSON outputs from GPT-5
        if _is_raw_tool_call_output(final_content):
            logger.info(f"Filtered out raw tool call JSON from non-streaming response")
            final_content = "検索を実行中です。しばらくお待ちください。"  # "Executing search. Please wait."
        
        # Convert LaTeX format for frontend compatibility
        final_content = _convert_latex_format(final_content)
        assistant_msg["content"] = final_content
        messages.append(assistant_msg)
        break
    # Store conversation with enhanced memory
    await store_conversation_messages_with_memory(conv_id, messages, str(user.id))

    # FORCE AI to generate response when tools were executed - don't just use fallback
    if (not final_content or not str(final_content).strip()) and tool_call_results:
        try:
            # Add a message to encourage AI response
            response_prompt = "Based on the tool results above, please provide a comprehensive response to the user's question in the appropriate language."
            messages.append({"role": "user", "content": response_prompt})
            
            retry_default = min(4000, MAX_TOKENS_PER_TURN // 2)
            final_retry_params = {
                "model": model_name,
                "messages": messages,
                "timeout": resolve_timeout(),
            }
            apply_token_limit(
                final_retry_params,
                resolve_max_tokens_with_cap(retry_default)
            )
            # Only include temperature if model supports it
            if not model_metadata.get("temperature_fixed"):
                final_retry_params["temperature"] = resolve_temperature()
                
            completion = await client.chat.completions.create(**final_retry_params)
            choice = (completion.choices or [None])[0]
            msg = getattr(choice, "message", None)
            content = getattr(msg, "content", "") if msg else ""
            if isinstance(content, list):
                content = "".join(_normalize_content_piece(p) for p in content)
            
            if content and content.strip():
                final_content = _convert_latex_format(content)
            
        except Exception as ai_retry_error:
            logger.error(f"Failed to force AI response in non-streaming: {ai_retry_error}")
    
    # If AI still didn't respond after retry, use fallback
    if (not final_content or not str(final_content).strip()) and tool_call_results:
        try:
            # Check if this is a Japanese request
            is_japanese = False
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if any(ord(c) > 0x3000 for c in content):  # Contains Japanese characters
                        is_japanese = True
                        break
            
            fallback_parts = []
            
            # Process all tool results, not just the last one
            for tool_result in tool_call_results:
                if not isinstance(tool_result, dict):
                    continue
                
                name = tool_result.get("name")
                res = tool_result.get("result") or {}
                
                if name == "get_company_profile" and isinstance(res, dict):
                    if is_japanese:
                        fallback_parts.append(_human_preview_company_profile_jp(res))
                    else:
                        fallback_parts.append(_human_preview_company_profile(res))
                elif name == "get_stock_quote" and isinstance(res, dict):
                    if is_japanese:
                        fallback_parts.append(_human_preview_quote_jp(res))
                    else:
                        fallback_parts.append(_human_preview_quote(res))
                elif name == "get_historical_prices" and isinstance(res, dict):
                    if is_japanese:
                        fallback_parts.append(_human_preview_historical_jp(res))
                    else:
                        fallback_parts.append(_human_preview_historical(res))
            
            # Combine all parts or use generic fallback
            if fallback_parts:
                final_content = _convert_latex_format("\n\n".join(fallback_parts))
            else:
                # Generic JSON fallback for unhandled tools
                try:
                    last = tool_call_results[-1] if tool_call_results else None
                    final_content = _convert_latex_format(
                        _safe_tool_result_json(last.get("result"))[:400] + "..."
                    )
                except Exception:
                    final_content = "(tool result received)"
        except Exception as e:
            logger.error(f"Non-streaming fallback generation error: {e}")
            pass

    # Use final content as the enhanced content (already converted in the loop above)
    enhanced_content = final_content or ""

    # Prepare response
    response = ChatResponse(
        content=enhanced_content or "", 
        tool_calls=tool_call_results or None, 
        conversation_id=conv_id
    )
    
    # Log tool usage for ML training (async, non-blocking)
    try:
        tools_called_names = [tc.get("name") for tc in tool_call_results if tc.get("name")]
        execution_time = time.time() - start_time
        log_tool_usage(
            query=req.prompt,
            tools_available=selected_tool_names,
            tools_called=tools_called_names,
            success=True,  # If we got here, request succeeded
            execution_time=execution_time,
            model=model_key,
            conversation_id=conv_id,
            user_id=str(user.id),
            tool_results=tool_call_results
        )
    except Exception as e:
        logger.error(f"Failed to log tool usage: {e}")
    
    # Cache the response for future identical requests (if new conversation)
    if not req.reset and not req.conversation_id:
        try:
            cache_response(req.prompt, model_key, sys_prompt, response.dict())
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
        finally:
            # Always clear in-flight marker
            clear_request_in_flight(req.prompt, model_key, sys_prompt)

    # Log action
    try:
        db.add(Log(
            user_id=int(user.id),
            action="chat",
            conversation_id=conv_id,
            prompt=req.prompt[:1000] if len(req.prompt) > 1000 else req.prompt,
            response=enhanced_content[:2000] if enhanced_content and len(enhanced_content) > 2000 else enhanced_content,
            tool_calls=tool_call_results
        ))
        db.commit()
    except Exception:
        db.rollback()

    return response

@router.post("/clear")
def chat_clear(payload: Dict[str, Any], _: User = Depends(get_current_user)):
    """Clear a conversation history."""
    conv_id = (payload.get("conversation_id") or "").strip()
    if not conv_id:
        raise HTTPException(status_code=400, detail="conversation_id required")
    
    existed = conv_clear(conv_id)
    return {"conversation_id": conv_id, "cleared": existed}

@router.get("/history/{conversation_id}", response_model=ChatHistoryResponse)
def chat_history(
    conversation_id: str, 
    include_system: bool = False, 
    max_messages: Optional[int] = None, 
    _: User = Depends(get_current_user)
):
    """Get conversation history."""
    msgs = conv_get(conversation_id)
    out: List[Dict[str, Any]] = []
    
    for m in msgs:
        role = m.get("role")
        if not include_system and role == "system":
            continue
        if role in {"user", "assistant", "system"}:
            out.append({"role": role, "content": m.get("content") or ""})
    
    if isinstance(max_messages, int) and max_messages > 0:
        out = out[-max_messages:]
    
    return ChatHistoryResponse(
        conversation_id=conversation_id, 
        found=bool(msgs), 
        messages=[ChatMessage(**mm) for mm in out]
    )

# Cleanup function for graceful shutdown
async def cleanup_chat_resources():
    """Cleanup chat-related resources like thread pools and connection pools."""
    try:
        # Shutdown thread pool
        _tool_executor.shutdown(wait=True, cancel_futures=False)
        logger.info("Tool executor thread pool shut down")
        
        # Cleanup connection pools
        await connection_pool.cleanup()
        logger.info("Connection pools cleaned up")
    except Exception as e:
        logger.error(f"Error during chat resource cleanup: {e}")
