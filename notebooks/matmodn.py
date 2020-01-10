# Adapted from https://stackoverflow.com/a/37015283
import numpy as np
from sympy import Matrix, Rational, mod_inverse, pprint

def mod(x,modulus):
    numer, denom = x.as_numer_denom()
    return numer*mod_inverse(denom,modulus) % modulus

def rref(mat, modulus):
    Ms = Matrix(mat)
    Ms_rref = Ms.rref(iszerofunc=lambda x: x % modulus==0)
    try:
        Ms_rref_fixed = Ms_rref[0].applyfunc(lambda x: mod(x,modulus))
    except ValueError:
        Ms_rref_fixed = Ms_rref[0]
    try:
        M_rref = np.asarray(Ms_rref_fixed, dtype=int)
    except ValueError:
        print(Ms_rref_fixed)
    return M_rref

def mod_rank(mat, modulus):
    M_rref = rref(mat, modulus)
    return np.linalg.matrix_rank(M_rref)

if __name__ == '__main__':
    A = np.asarray([
     [0, 0, 1, 1, 1, 1],
     [1, 1, 0, 0, 1, 1],
     [0, 1, 0, 0, 1, 0],
     [1, 0, 0, 1, 0, 0],
     [0, 0, 1, 1, 0, 0],
     [0, 1, 0, 1, 0, 0]])

    rref(A, 2)
    mod_rank(A, 2)
    np.linalg.matrix_rank(A)
