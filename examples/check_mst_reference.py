"""Check mathematical contracts of the single-head reference kernel."""
from mst_reference import fusion

query = [[1., 2.], [-1., 0.]]
key = [[1., 0.], [0., 1.], [-1., 1.]]
value = [[2., -3.], [4., 5.], [-2., 1.]]
support = [[True, False, True], [False, True, True]]
weights, output = fusion(query, key, value, support)
assert all(abs(sum(row)-1) < 1e-12 for row in weights)
assert all(weights[i][j] == 0 for i in range(2) for j in range(3)
           if not support[i][j])
for i in range(2):
    for d in range(2):
        legal = [value[j][d] for j in range(3) if support[i][j]]
        assert min(legal)-1e-12 <= output[i][d] <= max(legal)+1e-12
p, r = [1, 0], [2, 0, 1]
ap, zp = fusion([query[i] for i in p], [key[j] for j in r],
                [value[j] for j in r],
                [[support[i][j] for j in r] for i in p])
assert all(abs(zp[i][d]-output[p[i]][d]) < 1e-12
           for i in range(2) for d in range(2))
assert all(abs(ap[i][j]-weights[p[i]][r[j]]) < 1e-12
           for i in range(2) for j in range(3))
for args in [(query, key, value, [[False]*3, support[1]]),
             ([[float('nan'), 0.]], key, value, [[True]*3])]:
    try:
        fusion(*args)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid input accepted')
print('MST reference contracts passed (not a task-performance test).')
