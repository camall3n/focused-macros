from cube import cube, formula, skills

class primitive:
    alg_formulas = [[a] for a in cube.Action.keys()]
    actions = alg_formulas
    models = [cube.Cube().apply(a).summarize_effects() for a in actions]

class expert:
    alg_formulas = [
        skills.swap_3_edges_face,
        skills.swap_3_edges_mid,
        skills.swap_3_corners,
        skills.orient_2_edges,
        skills.orient_2_corners,
    ]
    options = [variation for f in alg_formulas for variation in formula.variations(f)]
    models = [cube.Cube().apply(o).summarize_effects() for o in options]

class random:
    alg_formulas = [skills.random_skill(len(a)) for a in expert.alg_formulas]
    options = [variation for f in alg_formulas for variation in formula.variations(f)]
    models = [cube.Cube().apply(o).summarize_effects() for o in options]

# class conjugates:
#     alg_formulas = [skills.random_conjugate(len(a)) for a in expert.alg_formulas]
#     options = [variation for f in alg_formulas for variation in formula.variations(f)]
#     models = [cube.Cube().apply(o).summarize_effects() for o in options]
#
# class commutators:
#     alg_formulas = [skills.random_commutator(len(a)) for a in expert.alg_formulas]
#     options = [variation for f in alg_formulas for variation in formula.variations(f)]
#     models = [cube.Cube().apply(o).summarize_effects() for o in options]
