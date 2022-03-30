
def _kullback(mata, matb):
    """Helper to check Kullback-Leibler distance."""
    mata = mata.flatten()
    matb = matb.flatten()
    return sum(a * (np.log(a) - np.log(b))
                  for a,b in zip(mata,matb))

def _entropy(orig :np.ndarray, reduc : np.ndarray) -> float:
    if sum(scipy.stats.entropy(orig, reduc)) != _kullback(orig, reduc):
        print(sum(scipy.stats.entropy(orig, reduc)), _kullback(orig, reduc))
        raise AssertionError('hola')
    return sum(scipy.stats.entropy(orig, reduc))

def _prosterior_kullback_diference(objectives, errors):
    cpd = bayesian_net.get_cpds(variable)
    original_values, reduced_values, time_, modified = _aproximate_cpd(cpd, error)

    if modified:
        modified_cpd = cpd.copy()
        modified_cpd.values = reduced_values

        modified_net = bayesian_net.copy()
        modified_net.add_cpds(modified_cpd)
    else:
        raise ValueError('This should not happen')
            

    # is left to compute posterior for both of them.
    prior_entropy = _entropy(original_values, reduced_values) 
    
    original_posterior_values = inference.VariableElimination(bayesian_net).query([variable]).values
    reduced_posterior_values = inference.VariableElimination(modified_net).query([variable]).values

    posterior_entropy = _entropy(original_posterior_values, reduced_posterior_values)
    if INTERACTIVE:
        print(f'\n\n*** Results for {_cpd_name(cpd)} in net {net_name}. ***\n')
        print(f'   - prior error: {prior_entropy}')
        print(f'   - posterior error: {posterior_entropy}')
            


def _posterior_kullback_diference_experiment(objectives, error):
    for error in errors:
        for net_name, goal_variables in objectives.items():
            bayesian_net = networks.get(net_name)

            for variable in goal_variables:

                
INTERACTIVE = True
VERBOSY = False
            
OBJECTIVE_NETS = {'hepar2.bif': ['ggtp', 'ast', 'alt', 'bilirubin'],
                  'diabetes.bif': ['cho_0'],
                  'munin.bif': ['L_MED_ALLCV_EW'],
                  'pathfinder.bif': ['F40']}


if __name__ == '__main__':
    errors =  [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

    _prosterior_kullback_diference(OBJECTIVE_NETS, errors)
