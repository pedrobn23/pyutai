import attr
import numpy as np
import typing
  
@attr.s 
class Cluster(object):
    indexes : set[int] = attr.ib() 
    weight : float = attr.ib()

    @classmethod
    def random(size = 5: int, *, sample_size = 40: int) -> Cluster:
        indexes = [random.randint(sample_size) for i in range size]
        weight = random.randint()
        return Cluster(indexes, weight)
        
    

class Potential:
    clusters :  set[Cluster] = attr.ib() 


if __name__ == '__main__':
    pot = Potential({Cluster({1,2,3}, 0.3), Cluster({4,5}, 0.25)}, Cluster({6,7,8,9}, 0.2)})

    
