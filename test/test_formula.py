import cube
from cube import formula

# Test mirroring
f = "R B' R' U' B' U F U' B U R B R' F'".split()
g = formula.mirror(f)
assert formula.mirror(g) == f
c = cube.Cube()
d = cube.Cube()
c.apply(f)
d.apply(g)
assert d == c


# Test simplify
f = "L R F F' R' U' D'".split()
s = formula.simplify(f)
assert s == "L U' D'".split()

f = "L D D'".split()
s = formula.simplify(f)
assert s == "L".split()

f = "L R F B U D L' R' F' B' U' D'".split()
s = formula.simplify(f + formula.inverse(f))
t = formula.simplify(formula.inverse(f) + f)
assert s == [] and t == []

f = "D' D'".split()
t = formula.simplify(f)
assert t == f, str(t)+" != "+str(f)

f = "L B' F' D D D D F B L".split()
s = formula.simplify(f)
assert s == ['L','L']

f = ['U', 'F', "L'", 'B', "U'", 'D', "L'", "U'", "U'", "D'", 'U', 'D', 'U', 'L', "D'", 'U', "B'", 'L', "F'", "U'"]
s = formula.simplify(f)
assert s == []

f = "L' R' L R".split()
s = formula.simplify(f)
assert s == []

f = "L' R' R' L R R".split()
s = formula.simplify(f)
assert s == []

f = ['D', 'D', 'U', 'D', 'D', "U'"]
s = formula.simplify(f)
assert s == []


# Test rotate
f = "R B' R' U' B' U F U' B U R B R' F'"
g = ' '.join(formula.rotate(f.split(), cube.Face.U))
h = ' '.join(formula.rotate(g.split(), cube.Face.D))
assert h==f
g = ' '.join(formula.rotate(f.split(), cube.Face.F, 2))
h = ' '.join(formula.rotate(f.split(), cube.Face.B, 2))
assert h==g
g = ' '.join(formula.rotate(f.split(), cube.Face.L, 3))
h = ' '.join(formula.rotate(f.split(), cube.Face.R, 1))
assert h==g
f = "R F B U L D".split()
g = formula.rotate(f, cube.Face.U, n=0)
assert f == g


# Test variations
f = ["R"]
c = ' '.join([' '.join(x) for x in sorted(formula.variations(f))])
actions = ' '.join(sorted(cube.actions))
assert c == actions

f = "R B' R' U' B' U F U' B U R B R' F'".split()
c = formula.variations(f)
assert len(c) == 96
