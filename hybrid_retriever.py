from rank_bm25 import BM25Okapi


def bm25_rerank(query, docs, top_k=8):

    if not docs:
        return []

    corpus = [
        doc.page_content.lower()
        for doc in docs
    ]

    tokenized_corpus = [
        text.split()
        for text in corpus
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(
        query.lower().split()
    )

    ranked_docs = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc
        for doc, score in ranked_docs[:top_k]
    ]