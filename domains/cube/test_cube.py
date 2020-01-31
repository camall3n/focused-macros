from domains.cube import Cube, pattern

def test_cube():
    """Test Cube functionality"""
    solved_cube = Cube()

    # Scrambles, effects
    other_cube = Cube()
    other_cube.scramble()
    assert other_cube != solved_cube
    assert len(other_cube.summarize_effects()) > 0

    # Applying a pattern
    other_cube = Cube()
    other_cube.apply(pattern.SUPERFLIP_QTM)
    assert other_cube != solved_cube

    # Applying the SUPERFLIP_QTM pattern twice restores the cube to its solved state
    other_cube.apply(pattern.SUPERFLIP_QTM)
    assert other_cube == solved_cube

    # Hashing and types
    assert hash(other_cube) == hash(solved_cube)
    assert isinstance(other_cube.summarize_effects(), tuple)
    assert other_cube.summarize_effects() == tuple()
    print('All tests passed.')

if __name__ == '__main__':
    test_cube()
