from cube import cube, formula, skills

alg_formulas = [
    skills.swap_3_edges_face,
    skills.swap_3_edges_mid,
    skills.swap_3_corners,
    skills.orient_2_edges,
    skills.orient_2_corners,
]

options = []
for alg in alg_formulas:
    variations = formula.variations(alg)
    options.extend(variations)

models = []
for o in options:
    m = cube.Cube().apply(o).summarize_effects()
    models.append(m)
