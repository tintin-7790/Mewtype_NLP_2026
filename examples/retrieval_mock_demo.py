"""Reproducible offline example; vectors are fixtures, not BGE-M3 output."""

from src.embedding_client import MockEmbeddingClient
from src.protocol import Case, Source
from src.retrieval import DenseEvidenceRetriever


def main():
    query = "入院时体温37.1℃"
    earlier = "1天前发热，体温最高38.5℃。"
    current = "入院时体温37.1℃。"
    case = Case("DEMO", "T0", [Source("S01", earlier, "现病史"), Source("S02", current, "体格检查")])
    client = MockEmbeddingClient({query: [1, 0], earlier: [0.4, 0.6], current: [0.99, 0.01]})
    retriever = DenseEvidenceRetriever(case, client)
    for candidate in retriever.retrieve(query, top_k=2):
        print(candidate.to_dict())
    print(retriever.stats())


if __name__ == "__main__":
    main()
