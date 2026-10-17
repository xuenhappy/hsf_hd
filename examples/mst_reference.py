from math import exp, isfinite, sqrt

def fusion(query, key, value, allowed):
    n, m = len(query), len(key)
    if not n or not m or len(value) != m:
        raise ValueError("empty or mismatched input")
    dk, dv = len(query[0]), len(value[0])
    if not dk or not dv or len(allowed) != n:
        raise ValueError("invalid dimensions")
    if any(len(row) != dk for row in query + key):
        raise ValueError("query/key width mismatch")
    if any(len(row) != dv for row in value):
        raise ValueError("value width mismatch")
    if any(len(row) != m for row in allowed):
        raise ValueError("support width mismatch")
    if any(type(a) is not bool for row in allowed for a in row):
        raise ValueError("support must be Boolean")
    if any(not isfinite(a) for row in query + key + value for a in row):
        raise ValueError("nonfinite input")
    weights, output = [], []
    for i, q in enumerate(query):
        support = [j for j in range(m) if allowed[i][j]]
        if not support:
            raise ValueError("query has no legal content")
        score = {j: sum(a*b for a, b in zip(q, key[j]))/sqrt(dk)
                 for j in support}
        if any(not isfinite(a) for a in score.values()):
            raise ValueError("nonfinite dot product")
        shift = max(score.values())
        numerator = {j: exp(score[j]-shift) for j in support}
        total = sum(numerator.values())
        row = [numerator.get(j, 0.0)/total for j in range(m)]
        z = [sum(row[j]*value[j][t] for j in support)
             for t in range(dv)]
        if any(not isfinite(a) for a in z):
            raise ValueError("nonfinite aggregation")
        weights.append(row)
        output.append(z)
    return weights, output
