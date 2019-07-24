import cube.cube as cube

c = cube.Cube()
d = cube.Cube()

c.scramble(30)
assert c != d
c.apply(cube.inverse_formula(c.formula))
assert c == d
