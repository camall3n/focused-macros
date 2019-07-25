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
f = "L R F F' R' U' D'".split()
s = cube.simplify_formula(f)
assert s == "L U' D'".split()

f = "L D D'".split()
s = cube.simplify_formula(f)
assert s == "L".split()

f = "L R F B U D L' R' F' B' U' D'".split()
s = cube.simplify_formula(f + cube.inverse_formula(f))
t = cube.simplify_formula(cube.inverse_formula(f) + f)
assert s == [] and t == []

f = "D' D'".split()
t = cube.simplify_formula(f)
assert t == f, str(t)+" != "+str(f)

f = "L B' F' D D D D F B L".split()
s = cube.simplify_formula(f)
assert s == ['L','L']

f = ['U', 'F', "L'", 'B', "U'", 'D', "L'", "U'", "U'", "D'", 'U', 'D', 'U', 'L', "D'", 'U', "B'", 'L', "F'", "U'"]
s = cube.simplify_formula(f)
assert s == []

f = "L' R' L R".split()
s = cube.simplify_formula(f)
assert s == []

f = "L' R' R' L R R".split()
s = cube.simplify_formula(f)
assert s == []

f = ['D', 'D', 'U', 'D', 'D', "U'"]
s = cube.simplify_formula(f)
assert s == []
