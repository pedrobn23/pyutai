def all(path):
    for net in os.listdir(path):
        if net.endswith('.bif'):
            file_ = read.read(f'networks/{net}')
            model = file_.get_model()
            cpds = model.get_cpds()
            for cpd in cpds:
                yield cpd


def smalls(path, *, threshold=3000):
    cardinalities = set()
    for cpd in  all(path):
     	 if utils.unique_values(cpd) < threshold) and 
