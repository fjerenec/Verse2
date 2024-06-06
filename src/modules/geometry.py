import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

import modules.data as data
import numpy as np

class Geometry():
    def __init__(self) -> None:
        self.sets = data.Sets()

    def input_part_nodes(self, inputCoordinates, inputVolumes) -> None:
        self.part_nodes = _PartNodes(arrayOfNodeCoords = inputCoordinates, arrayOfNodeVolumes = inputVolumes)

class _PartNodes():
    #Only one instance of this should ever be created in a single NumericalModel, since it contains the original nodes that constitute the part!
    def __init__(self, arrayOfNodeCoords: np.ndarray[float,2], arrayOfNodeVolumes: np.ndarray[float,1],dim:int = 2) -> None:
        self.partNodesTable = {}
        self.coordVec = arrayOfNodeCoords
        self.dim = dim
        for i in range(arrayOfNodeCoords.shape[0]):
            self.partNodesTable[i] = _Node(nodeID = i, coordinatesArray = arrayOfNodeCoords[i], volume = arrayOfNodeVolumes[i])

    def get_part_nodes(self):
        return self.partNodesTable
 
    # def delete_part_nodes(self, partNodeIDs: np.ndarray[int,1]) -> None:
    #     for i in partNodeIDs:
    #         if i in self.partNodesTable.keys():
    #             del self.partNodesTable[i]
    #         else:
    #             print("Node ID does not exist. Cannot delete nonexistent PartNode")


    def add_part_nodes(self, arrayOfAddedNodesCoords: np.ndarray[float,2], addedNodesVolume: np.ndarray[float,1]) -> None:
        #Node Ids start at 0 (python indexing). If len is 10 the max ID in the array would be 9 (0 to 9 are 10 numbers!)
        numOfCurPartNodes = len(self.partNodesTable)
        for i in range(arrayOfAddedNodesCoords.shape[0]):
            if i in self.partNodesTable.keys():
                print("ID of added PartNode already exists")
            else:
                newNodeID = numOfCurPartNodes + i
                self.partNodesTable[newNodeID] = _Node(nodeID = newNodeID, coordinatesArray = arrayOfAddedNodesCoords[i], volume = addedNodesVolume[i])

class _Node():
    def __init__(self,nodeID:int, coordinatesArray: np.ndarray[float,1], volume: float) -> None:
        self.__nodeID = nodeID
        self.__coordinates = coordinatesArray
        self.__volume = volume

    def ID(self) -> int:
        return self.__nodeID
    
    def coords(self) -> np.ndarray[float,1]:
        return self.__coordinates

    def vol(self) -> float:
        return self.__volume 

    def all_data(self) -> list[int,list[float,2],float]:
        """
        Returns [nodeID, list[coordinates], volume]
        """
        return [self.__nodeID, list(self.__coordinates),self.__volume]
