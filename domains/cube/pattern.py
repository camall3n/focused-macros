import random
from domains.cube.skills import random_skill

cube_in_cube = "F L F U' R U F F L L U' L' B D' B' L L U".split()
exchanged_rings = "B' U' B' L' D B U D D B U L D' L' U' L L D".split()
twisted_peaks = "F B' U F U F U L B L L B' U F' L U L' B".split()
exchanged_peaks = "F U U L F L' B L U B' R' L' U R' D' F' B R R".split()
twisted_crosses = "R R L' D F F R' D' R' L U' D R D B B R' U D D".split()
six_spots = "U D' R L' F B' U D'".split()
scramble1 = "L' B' L R' U L' B' F' L B L F' B F B U' R' R F B' F' U F' F' L' B B' L D B' L U' R' F' B R B' D' L' D L' F R D' R L F' R' R U B F' L' B' F' R' L B' D B".split()

def scramble(seed=0, length=60):
    st = random.getstate()
    random.seed(seed)
    formula = random_skill(length)
    random.setstate(st)
    return formula

def main():
    assert scramble(1) == scramble(1)
    assert scramble(1) != scramble(2)
    from domains.cube import formula
    assert ' '.join(formula.simplify(scramble(1))) == "U D' R F R' R' F' L' D L' D' L B' R' F B' R B L' F' U' F' D L' B' L U' D R' R' U' D B D F' D R' F U' F' R U B' R"


if __name__ == '__main__':
    main()
