def build_context(retrieval_result: dict) -> str:
    chunks = retrieval_result.get("vector_chunks", [])
    graph_ctx = retrieval_result.get("graph_context", "")

    seen, unique_chunks = set(), []

    for c in chunks:
        key = c["text"][:100]
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    graph_section = ""
    if graph_ctx:
        graph_section = "=== KNOWLEDGE GRAPH RELATIONSHIPS === \n"
        graph_section += graph_ctx + "\n\n"

    chunk_section = "=== RELEVANT DOCUMENT CHUNKS ===\n"
    for i, c in enumerate(unique_chunks, 1):
        chunk_section += (
            f"[{i}] Source: {c['filename']} | Score: {c['score']}\n"
            f"{c['text'][:800]}\n\n"
        )
    
    full_context = (graph_section + chunk_section).strip()
    return full_context[:6000]

def build_citation_list(vector_chunks: list) -> list[str]:
    return list({c["filename"] for c in vector_chunks})

    