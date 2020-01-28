import cube
from cube import formula

c = cube.Cube()
d = cube.Cube()

# Test scramble and inverses
c.scramble(30)
assert c != d
assert len(c.summarize_effects()) > 0
c.apply(formula.inverse(c.sequence))
assert c == d

assert hash(c) == hash(d)
assert c.summarize_effects() == tuple()
