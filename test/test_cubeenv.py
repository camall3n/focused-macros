from cube.cubeenv import CubeEnv

c = CubeEnv()
c.scramble(5)
c.render()
print(c.formula)
