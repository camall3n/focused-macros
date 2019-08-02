from cube import cube, formula, skills, expert

random_formulas = [skills.random_skill(len(a)) for a in expert.alg_formulas]
conjugate_formulas = [skills.random_conjugate(len(a)) for a in expert.alg_formulas]
commutator_formulas = [skills.random_commutator(len(a)) for a in expert.alg_formulas]

random_options = [variation for f in random_formulas for variation in formula.variations(f)]
conjugate_options = [variation for f in conjugate_formulas for variation in formula.variations(f)]
commutator_options = [variation for f in commutator_formulas for variation in formula.variations(f)]

random_models = [cube.Cube().apply(o).summarize_effects() for o in random_options]
conjugate_models = [cube.Cube().apply(o).summarize_effects() for o in conjugate_options]
commutator_models = [cube.Cube().apply(o).summarize_effects() for o in commutator_options]
