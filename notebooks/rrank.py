import numpy as np

def reduce(M):
    n = len(M)
    for col in range(n):
        M = sorted(map(tuple,list(M)),reverse=True)
        keep_row = M[col]
        for row in range(col+1,n):
            if M[row][col] == 1:
                M[row] = tuple((np.array(M[row]) + np.array(keep_row))%2)
    for col in reversed(range(n)):
        keep_row = M[col]
        for row in range(col):
            if np.all(np.array(M[row]) * np.array(keep_row) == np.array(keep_row)):
                M[row] = tuple((np.array(M[row]) + np.array(keep_row))%2)
    return np.stack(M)

def rrank(M):
    M = reduce(M)
    return np.linalg.matrix_rank(M)

def test():
    A = np.asarray([
     [0, 0, 1, 1, 1, 1],
     [1, 1, 0, 0, 1, 1],
     [0, 1, 0, 0, 1, 0],
     [1, 0, 0, 1, 0, 0],
     [0, 0, 1, 1, 0, 0],
     [0, 1, 0, 1, 0, 0]])

    reduce(A)
    assert rrank(A) == 5

    M = np.array([
           [0, 1, 1, 0, 0, 1, 1, 1, 0, 1],
           [0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
           [1, 1, 1, 0, 1, 0, 0, 0, 1, 1],
           [1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
           [1, 0, 1, 1, 1, 1, 0, 0, 1, 0],
           [0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
           [0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
           [0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 0, 0, 0, 0, 1]])
    reduce(M)
    assert rrank(M) == 9

if __name__ == '__main__':
    test()
    print('All tests passed.')
