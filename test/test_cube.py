import cube.cube as cube

c = cube.Cube()
d = cube.Cube()

# Test scramble and inverses
c.scramble(30)
assert c != d
c.apply(cube.inverse_formula(c.formula))
assert c == d

# Test mirroring
f = "R B' R' U' B' U F U' B U R B R' F'".split()
g = cube.mirror_formula(f)
assert cube.mirror_formula(g) == f
c = cube.Cube()
d = cube.Cube()
c.apply(f)
d.apply(g)
assert d == c

# Test simplify
f = "L R F B U D L' R' F' B' U' D'".split()
s = cube.simplify_formula(f + cube.inverse_formula(f))
assert s == []
