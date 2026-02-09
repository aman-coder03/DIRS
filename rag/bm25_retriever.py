from rank_bm25 import BM25Okapi
import numpy as np
import re

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

class BM25Retriever:
    def __init__(self, documents, tokenized_docs=None):
        self.documents = documents

        if tokenized_docs is not None:
            self.tokenized_docs = tokenized_docs
        else:
            self.tokenized_docs = [tokenize(doc) for doc in documents]

        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query, top_k=5):
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.documents[i], scores[i]) for i in top_indices]
