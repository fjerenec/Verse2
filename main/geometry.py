import data

class Geometry():
    def __init__(self) -> None:
        self.sets = data.Sets()

    def input_part_nodes(self, inputCoordinates, inputVolumes) -> None:
        self.part_nodes = data.PartNodes(arrayOfNodeCoords = inputCoordinates, arrayOfNodeVolumes = inputVolumes)
