from cube import cube, formula, skills

alg_formulas = [
    skills.swap_3_edges_face,
    skills.swap_3_edges_mid,
    skills.swap_3_corners,
    skills.orient_2_edges,
    skills.orient_2_corners,
]

options = [variation for f in alg_formulas for variation in formula.variations(f)]
models = [cube.Cube().apply(o).summarize_effects() for o in options]
