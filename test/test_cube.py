from cube import cube
from cube import formula

c = cube.Cube()
d = cube.Cube()

# Test scramble and inverses
c.scramble(30)
assert c != d
c.apply(formula.inverse(c.sequence))
assert c == d

assert hash(c) == hash(d)
