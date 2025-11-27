"""RAG (Retrieval-Augmented Generation) service for knowledge base management with performance optimizations."""
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from functools import lru_cache
from app.core.config import (
    RAG_ENABLED, KNOWLEDGE_DIR, CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT, OPENAI_EMBEDDINGS_MODEL,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    OPENAI_API_KEY, OPENAI_BASE_URL, _normalize_azure_endpoint
)
from app.services.openai_client import get_provider

logger = logging.getLogger(__name__)

# Global caches for performance
_embeddings_cache = None
_vectorstore_cache = None
_last_index_check = 0

def _ensure_dirs():
    """Ensure knowledge and chroma directories exist with proper permissions."""
    try:
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        os.makedirs(CHROMA_DIR, exist_ok=True)
        
        # Ensure directories are writable
        os.chmod(KNOWLEDGE_DIR, 0o755)
        os.chmod(CHROMA_DIR, 0o755)
        
        # If .chroma has any files, ensure they're writable too
        for root, dirs, files in os.walk(CHROMA_DIR):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o755)
            for f in files:
                os.chmod(os.path.join(root, f), 0o644)
                
    except Exception as e:
        logger.warning(f"Could not ensure directory permissions: {e}")
        pass

@lru_cache(maxsize=1)
def _get_embeddings():
    """Return a cached LangChain embeddings instance for the active provider."""
    global _embeddings_cache

    if _embeddings_cache is not None:
        return _embeddings_cache

    if not RAG_ENABLED:
        raise RuntimeError("RAG is not enabled")
    
    try:
        # Lazy import to avoid hard dependency if RAG_DISABLED
        from importlib import import_module

        # Prefer OpenAI embeddings when available, even if Azure is configured
        provider = "openai" if OPENAI_API_KEY else get_provider()
        if provider not in {"azure", "openai"}:
            provider = "openai" if OPENAI_API_KEY else "azure"

        if provider == "azure":
            # langchain-openai Azure embeddings
            AzureEmb = import_module("langchain_openai").AzureOpenAIEmbeddings
            if not AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT:
                raise RuntimeError("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT is required for Azure RAG")
            # Configure for text-embedding-3-large model
            embedding_config = {
                "azure_deployment": AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
                "api_key": AZURE_OPENAI_API_KEY,
                "azure_endpoint": _normalize_azure_endpoint(AZURE_OPENAI_ENDPOINT),
                "openai_api_version": AZURE_OPENAI_API_VERSION,
                # Performance optimizations
                "chunk_size": 50,  # Smaller batches for large model
                "max_retries": 2,   # Extra retry for reliability  
                "request_timeout": 45  # Longer timeout for large embeddings
            }
            
            # Add dimensions parameter for text-embedding-3-large
            # Use 1536 for compatibility with existing indexes or 3072 for full quality
            if "text-embedding-3-large" in str(AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT).lower():
                embedding_config["dimensions"] = 1536  # Reduced for compatibility
                logger.info("Using text-embedding-3-large with 1536 dimensions for compatibility")
            
            _embeddings_cache = AzureEmb(**embedding_config)
        else:
            # Standard OpenAI embeddings
            OpenEmb = import_module("langchain_openai").OpenAIEmbeddings
            kwargs = {
                "model": OPENAI_EMBEDDINGS_MODEL,
                "api_key": OPENAI_API_KEY,
                "chunk_size": 50,  # Smaller batches for large model
                "max_retries": 2,   # Extra retry for reliability
                "request_timeout": 45  # Longer timeout for large embeddings
            }
            
            # Add dimensions parameter for text-embedding-3-large
            if "text-embedding-3-large" in OPENAI_EMBEDDINGS_MODEL.lower():
                kwargs["dimensions"] = 1536  # Reduced for compatibility
                logger.info("Using text-embedding-3-large with 1536 dimensions for compatibility")
            
            if (OPENAI_BASE_URL or "").strip():
                kwargs["base_url"] = OPENAI_BASE_URL.strip()
            _embeddings_cache = OpenEmb(**kwargs)

        logger.info(f"Embeddings client cached for provider: {provider}")
        return _embeddings_cache

    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        raise

def rag_reindex(clear: bool = True) -> Dict[str, Any]:
    """(Re)index all knowledge base files into Chroma with performance optimizations."""
    global _vectorstore_cache, _embeddings_cache, _last_index_check

    if not RAG_ENABLED:
        return {"enabled": False}
    
    _ensure_dirs()

    # Check if embedding model changed - if so, force clear to avoid dimension mismatch
    embedding_model_file = os.path.join(CHROMA_DIR, ".embedding_model")
    # Keep model marker aligned with the provider preference (OpenAI first)
    if OPENAI_API_KEY:
        current_model = OPENAI_EMBEDDINGS_MODEL
    elif AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT:
        current_model = AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT
    else:
        current_model = OPENAI_EMBEDDINGS_MODEL
    force_clear = clear
    
    if os.path.exists(embedding_model_file) and not clear:
        try:
            with open(embedding_model_file, 'r') as f:
                stored_model = f.read().strip()
            if stored_model != current_model:
                logger.info(f"Embedding model changed from {stored_model} to {current_model} - forcing clear reindex")
                force_clear = True
        except:
            force_clear = True
    
    # Clear caches when reindexing
    _vectorstore_cache = None
    _embeddings_cache = None
    _get_embeddings.cache_clear()

    # Lazy imports with error handling
    try:
        from importlib import import_module
        TextSplitter = import_module("langchain.text_splitter").RecursiveCharacterTextSplitter
        Document = import_module("langchain_core.documents").Document
        # Try different possible chroma module names
        try:
            Chroma = import_module("langchain_chroma").Chroma
        except ImportError:
            try:
                Chroma = import_module("langchain_community.vectorstores.chroma").Chroma
            except ImportError:
                try:
                    Chroma = import_module("langchain.vectorstores.chroma").Chroma
                except ImportError:
                    logger.error("No compatible Chroma vectorstore module found")
                    return {"enabled": False, "error": "Chroma vectorstore not available"}
    except ImportError as e:
        logger.error(f"Missing required packages for RAG: {e}")
        return {"enabled": False, "error": "Missing dependencies"}

    # Optionally clear existing index
    if force_clear:
        try:
            import shutil
            if os.path.isdir(CHROMA_DIR):
                # Ensure permissions before removal
                try:
                    os.chmod(CHROMA_DIR, 0o755)
                    for root, dirs, files in os.walk(CHROMA_DIR):
                        for d in dirs:
                            os.chmod(os.path.join(root, d), 0o755)
                        for f in files:
                            os.chmod(os.path.join(root, f), 0o644)
                except:
                    pass
                shutil.rmtree(CHROMA_DIR, ignore_errors=True)
            os.makedirs(CHROMA_DIR, exist_ok=True)
            os.chmod(CHROMA_DIR, 0o755)
        except Exception as e:
            logger.warning(f"Could not clear chroma directory: {e}")
            pass

    # Store current embedding model for future compatibility checks
    try:
        with open(embedding_model_file, 'w') as f:
            f.write(current_model)
    except Exception as e:
        logger.warning(f"Could not save embedding model info: {e}")

    # Load files (.txt, .md) with size limits for performance
    docs: List[Any] = []
    kb_path = Path(KNOWLEDGE_DIR)
    exts = {".txt", ".md", ".mdx"}
    max_file_size = 1024 * 1024  # 1MB limit per file

    for p in kb_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                # Check file size first
                if p.stat().st_size > max_file_size:
                    logger.warning(f"Skipping large file {p.name} ({p.stat().st_size} bytes)")
                    continue

                content = p.read_text(encoding="utf-8", errors="ignore")
                if content and content.strip():
                    # Limit content length for performance
                    if len(content) > 50000:  # 50k chars max
                        content = content[:50000] + "... [truncated]"
                    docs.append(Document(page_content=content, metadata={"path": str(p)}))
            except Exception as e:
                logger.warning(f"Error loading {p}: {e}")
                continue

    if not docs:
        # create empty collection
        try:
            _ensure_dirs()
            vs = Chroma(
                persist_directory=CHROMA_DIR,
                collection_name="knowledge_base",
                embedding_function=_get_embeddings()
            )
            _vectorstore_cache = vs
            _last_index_check = os.path.getmtime(CHROMA_DIR) if os.path.exists(CHROMA_DIR) else 0
            return {"enabled": True, "files": 0, "chunks": 0}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to create empty vectorstore: {error_msg}")
            
            if "readonly database" in error_msg:
                logger.error("ChromaDB permission issue - trying to fix...")
                try:
                    import shutil
                    if os.path.exists(CHROMA_DIR):
                        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
                    _ensure_dirs()
                    
                    vs = Chroma(
                        persist_directory=CHROMA_DIR,
                        collection_name="knowledge_base",
                        embedding_function=_get_embeddings()
                    )
                    _vectorstore_cache = vs
                    _last_index_check = os.path.getmtime(CHROMA_DIR) if os.path.exists(CHROMA_DIR) else 0
                    logger.info("Empty RAG collection created after permission fix")
                    return {"enabled": True, "files": 0, "chunks": 0}
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}")
                    return {"enabled": False, "error": f"Permission issue: {error_msg}"}
            
            return {"enabled": False, "error": error_msg}

    # Optimize chunking for performance
    splitter = TextSplitter(
        chunk_size=min(CHUNK_SIZE, 800),  # Smaller chunks for faster processing
        chunk_overlap=min(CHUNK_OVERLAP, 100),
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # Better splitting
    )

    try:
        chunks = splitter.split_documents(docs)

        # Limit total chunks for performance
        if len(chunks) > 1000:  # Limit to 1000 chunks
            logger.warning(f"Limiting chunks from {len(chunks)} to 1000 for performance")
            chunks = chunks[:1000]

        # Ensure directory exists and is writable before creating vectorstore
        _ensure_dirs()
        
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=_get_embeddings(),
            persist_directory=CHROMA_DIR,
            collection_name="knowledge_base",
        )

        _vectorstore_cache = vs
        _last_index_check = os.path.getmtime(CHROMA_DIR) if os.path.exists(CHROMA_DIR) else 0

        logger.info(f"RAG index created with {len(chunks)} chunks from {len(docs)} files")
        return {"enabled": True, "files": len(docs), "chunks": len(chunks)}

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to create vectorstore: {error_msg}")
        
        # Try to provide a more helpful error message
        if "readonly database" in error_msg:
            logger.error("ChromaDB database permission issue. Trying to fix...")
            try:
                # Try to fix permissions and retry once
                import shutil
                if os.path.exists(CHROMA_DIR):
                    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
                _ensure_dirs()
                
                vs = Chroma.from_documents(
                    documents=chunks,
                    embedding=_get_embeddings(),
                    persist_directory=CHROMA_DIR,
                    collection_name="knowledge_base",
                )
                
                _vectorstore_cache = vs
                _last_index_check = os.path.getmtime(CHROMA_DIR) if os.path.exists(CHROMA_DIR) else 0
                logger.info(f"RAG index created after permission fix with {len(chunks)} chunks from {len(docs)} files")
                return {"enabled": True, "files": len(docs), "chunks": len(chunks)}
                
            except Exception as retry_e:
                logger.error(f"Retry also failed: {retry_e}")
                return {"enabled": False, "error": f"Permission issue: {error_msg}"}
        
        return {"enabled": False, "error": error_msg}

def _get_vectorstore():
    """Get the cached Chroma vectorstore instance with lazy loading."""
    global _vectorstore_cache, _last_index_check

    if not RAG_ENABLED:
        return None

    # Check if we need to reload (index changed)
    current_mtime = os.path.getmtime(CHROMA_DIR) if os.path.exists(CHROMA_DIR) else 0
    if _vectorstore_cache is None or current_mtime > _last_index_check:
        try:
            from importlib import import_module
            # Try different possible chroma module names
            try:
                Chroma = import_module("langchain_chroma").Chroma
            except ImportError:
                try:
                    Chroma = import_module("langchain_community.vectorstores.chroma").Chroma
                except ImportError:
                    try:
                        Chroma = import_module("langchain.vectorstores.chroma").Chroma
                    except ImportError:
                        logger.error("No compatible Chroma vectorstore module found in _get_vectorstore")
                        return None

            _ensure_dirs()

            # Check if index exists
            if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
                logger.info("No RAG index found, creating empty one")
                return None

            _vectorstore_cache = Chroma(
                persist_directory=CHROMA_DIR,
                collection_name="knowledge_base",
                embedding_function=_get_embeddings()
            )
            _last_index_check = current_mtime
            logger.debug("Vectorstore loaded from cache")

        except Exception as e:
            logger.error(f"Failed to load vectorstore: {e}")
            return None

    return _vectorstore_cache

def rag_search(query: str, k: int = 4) -> Dict[str, Any]:
    """Retrieve top-k relevant knowledge chunks to ground answers (RAG)."""
    if not RAG_ENABLED:
        return {"enabled": False, "results": []}
    if not (query or "").strip():
        return {"enabled": True, "query": query, "count": 0, "results": []}
    
    try:
        vs = _get_vectorstore()
        if not vs:
            return {"enabled": False, "results": []}
        
        # Use score-enabled search if available
        items = []
        try:
            results = vs.similarity_search_with_score(query, k=max(1, int(k)))
            for doc, score in results:
                items.append({
                    "text": doc.page_content, 
                    "metadata": doc.metadata, 
                    "score": float(score) if score is not None else None
                })
        except Exception:
            results = vs.similarity_search(query, k=max(1, int(k)))
            for doc in results:
                items.append({"text": doc.page_content, "metadata": doc.metadata})
        
        return {
            "enabled": True, 
            "query": query, 
            "k": int(k), 
            "count": len(items), 
            "results": items
        }
    except Exception as e:
        return {"enabled": True, "query": query, "error": str(e), "results": []}

def get_rag_status() -> Dict[str, Any]:
    """Get RAG system status and configuration."""
    try:
        kb_path = Path(KNOWLEDGE_DIR)
        files = list(kb_path.glob("*.md")) + list(kb_path.glob("*.txt")) if kb_path.exists() else []

        chroma_path = Path(CHROMA_DIR)
        indexed = chroma_path.exists() and any(chroma_path.iterdir()) if chroma_path.exists() else False

        return {
            "enabled": RAG_ENABLED,
            "knowledge_dir": KNOWLEDGE_DIR,
            "chroma_dir": CHROMA_DIR,
            "knowledge_files": [f.name for f in files],
            "knowledge_file_count": len(files),
            "indexed": indexed,
            "azure_embeddings_deployment": AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        }
    except Exception as e:
        logger.error(f"RAG status error: {e}")
        return {"enabled": False, "error": str(e)}
