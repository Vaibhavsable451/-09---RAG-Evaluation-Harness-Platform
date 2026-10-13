def hit_at_k(ranked_ids, relevant, k):
    return float(any(r in relevant for r in ranked_ids[:k]))

def recall_at_k(ranked_ids, relevant, k):
    return len(set(ranked_ids[:k]) & set(relevant)) / len(relevant)

def reciprocal_rank(ranked_ids, relevant):
    for i, r in enumerate(ranked_ids, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0
