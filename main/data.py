import numpy as np

class PartNodes():
    #Only one instance of this should ever be created in a single NumericalModel, since it contains the original nodes that constitute the part!
    def __init__(self, arrayOfNodeCoords: np.ndarray[float,2], arrayOfNodeVolumes: np.ndarray[float,1]) -> None:
        self.partNodesTable = {}
        for i in range(arrayOfNodeCoords.shape[0]):
            self.partNodesTable[i] = Node(nodeID = i, coordinatesArray = arrayOfNodeCoords[i], volume = arrayOfNodeVolumes[i])

    def get_part_nodes(self):
        return self.partNodesTable
 
    def delete_part_nodes(self, partNodeIDs: np.ndarray[int,1]) -> None:
        for i in partNodeIDs:
            if i in self.partNodesTable.keys():
                del self.partNodesTable[i]
            else:
                print("Node ID does not exist. Cannot delete nonexistent PartNode")

    def add_part_nodes(self, arrayOfAddedNodesCoords: np.ndarray[float,2], addedNodesVolume: np.ndarray[float,1]) -> None:
        #Node Ids start at 0 (python indexing). If len is 10 the max ID in the array would be 9 (0 to 9 are 10 numbers!)
        numOfCurPartNodes = len(self.partNodesTable)
        for i in range(arrayOfAddedNodesCoords.shape[0]):
            if i in self.partNodesTable.keys():
                print("ID of added PartNode already exists")
            else:
                newNodeID = numOfCurPartNodes + i
                self.partNodesTable[newNodeID] = Node(nodeID = newNodeID, coordinatesArray = arrayOfAddedNodesCoords[i], volume = addedNodesVolume[i])

class Node():
    def __init__(self,nodeID:int, coordinatesArray: np.ndarray[float,1], volume: float) -> None:
        self.nodeID = nodeID
        self.coordinates = coordinatesArray
        self.volume = volume

    def ID(self) -> int:
        return self.nodeID
    
    def coords(self) -> np.ndarray[float,1]:
        return self.coordinates

    def vol(self) -> float:
        return self.volume 

    def all_data(self) -> list[int,list[float,2],float]:
        return [self.nodeID, list(self.coordinates),self.volume]

class Sets():
    #Sets only contain the ID of the wanted type stored in the set and not the actual type itself!
    def __init__(self) -> None:
        #Initiate the hashtable for each of the "Set" subclasses
        self.setTable = {}

    def create_set(self,arrayOfIDs: np.ndarray[int,1], setName: str = "set name") -> None:
        #Need to check if keyword already exists. If so i need to warn the user if he want sto rewrite the data!
        if arrayOfIDs.dtype != int:
            print("All data in arrayOfIDs must of of type np.int64!")
            return 
        
        if type(setName) != str:
            setName = str(setName)

        if setName in self.setTable.keys():
            print(f"A set with name -{setName}- already exists. Would you like to overwrite?")
            override = get_user_decision()
            if override == True:
                print("Set overwritten")
                newSet = Set(arrayOfIDs, name = setName)
                self.setTable[setName] = newSet
            else:
                return
        else:
            newSet = Set(arrayOfIDs, name = setName)
            self.setTable[setName] = newSet
        
    def delete_set(self,setName: str = "set name"):
        if setName in self.sets.keys():
            del self.setTable[setName]
        else:
            print("No set with given name exists. Nonexistent set cannot be deleted!")

class Set():
    def __init__(self,arrayOfIDs: np.ndarray[int,1], name: str = "Set") -> None:
        self.name = name
        self.IDarray = arrayOfIDs
    def type(self) -> type:

        return type(self)
    
    def get_data(self):
        return self.IDarray[:]

def get_user_decision():
    while True:
        decision = input("Please enter your decision (Y/N): ").upper()  # Convert input to uppercase
        if decision in ('Y', 'N'):
            if decision == "Y":
                return True
            else:
                return False
        else:
            print("Invalid input. Please enter either 'Y' or 'N'.")