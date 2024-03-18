class Geometry():
    def __init__(self) -> None:
        sets = Sets()
        pass

class Sets():
    def __init__(self) -> None:
        #Initiate the hashtable for each of the "Set" subclasses
        self.nodeSets = {}
        self.bondSets = {}

    def add_node_set(self,name: str = "set name") -> None:
        #Need to check if keyword already exists. If so i need to warn the user if he want sto rewrite the data!
        newNodeSet = NodeSet(name = name)
        self.nodeSets[name] = newNodeSet

class Set():
    
    def __init__(self,name: str = "Set") -> None:
        self.name = name
        
class NodeSet(Set):
    def __init__(self, name: str = "NodeSet") -> None:
        super().__init__(name)

class BondSet(Set):
    def __init__(self, name: str = "BondSet") -> None:
        super().__init__(name)
