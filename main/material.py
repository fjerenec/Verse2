from data import get_user_decision
from data import Set
class Materials():
    def __init__(self) -> None:
        self.materialsTable = {}
        self.materialsByIDTable = {material.materialID: material for material in self.materialsTable.values()}

    def create_material(self,materialName: str = "MaterialName", isIsotropic:bool = True):
        if type(materialName) != str:
            materialName = str(materialName)
        
        if materialName in self.materialsTable.keys():
            print(f"A material with name -{materialName}- already exists. Would you like to overwrite?")
            override = get_user_decision()
            if override == True:
                print("Material overwritten")
                if isIsotropic:
                    self.materialsTable[materialName] = Isotropic(materialName=materialName)
                else:
                    self.materialsTable[materialName] = Anisotropic(materialName=materialName)
                
                self.materialsByIDTable[self.materialsTable[materialName].materialID] = self.materialsTable[materialName]
            else:
                return
        else:
            if isIsotropic:
                self.materialsTable[materialName] = Isotropic(materialName=materialName)
            else:
                self.materialsTable[materialName] = Anisotropic(materialName=materialName)
            self.materialsByIDTable[self.materialsTable[materialName].materialID] = self.materialsTable[materialName]


    def delete_material(self, materialName: str)-> None:
        if type(materialName) != str:
            materialName = str(materialName)

        if materialName in self.materialsTable.keys():
            self.materialsTable[materialName].delete_material_id()
            del self.materialsTable[materialName]


class Material():
    next_id = 1

    deleted_ids = set()

    def __init__(self,materialName:str = "Material") -> None:
        self.name = materialName

        if Material.deleted_ids:
            self.materialID = Material.deleted_ids.pop()
        else:
            self.materialID = Material.next_id 
            Material.next_id += 1

    def delete_material_id(self) -> None:
        Material.deleted_ids.add(self.materialID)

    def set_density(self,density:float):
        self.density = density

    def set_stretch_thresholds(self,damageInitStretch: float, breakageStretch: float):
        self.s0 = damageInitStretch
        self.sc = breakageStretch

    def get_material_properties(self):
        return vars(self)       
    
class Isotropic(Material):
    def __init__(self,materialName: str = "Material") -> None:
        super().__init__(materialName)

    def set_tensile_properties(self,youngMod: float,poisson:float):
        self.Emod = youngMod
        self.poisson = poisson


class Anisotropic(Material):
    def __init__(self, materialName: str = "Material") -> None:
        super().__init__(materialName)

    def set_tensile_properties(self,youngModx: float ,youngMody: float ,youngModz: float,poisx:float, poisy:float,poisz:float):
        self.Emodx = youngModx
        self.Emody = youngMody
        self.Emodz = youngModz
        self.poisx = poisx
        self.poisy = poisy
        self.poisz = poisz

class MaterialSections():
    def __init__(self) -> None:
        self.materialSectionsTable = {}

    def create(self,nodeSet: Set, material: Material, name:str ):
        if name in self.materialSectionsTable.keys():
            print(f"A material section with name -{name}- already exists. Would you like to overwrite?")
            override = get_user_decision()
            if override == True:
                print("Material section overwritten")
                self.materialSectionsTable[name] = MaterialSection(nodeSet=nodeSet, material=material, name = name)
        else:
            self.materialSectionsTable[name] = MaterialSection(nodeSet=nodeSet, material=material, name = name)

class MaterialSection():

    def __init__(self,nodeSet:Set, material: Material, name:str = "MaterialSection") -> None:
        self.nodeSet = nodeSet
        self.material = material
        self.name = name

class MaterialInterfaces():
    def __init__(self) -> None:
        self.materialInterfacesTable = {}
    
    def create(self,materialSection1: MaterialSection, materialSection2: MaterialSection, material: Material, name:str = "MaterialInterface"):
        if name in self.materialInterfacesTable.keys():
            print(f"A material interface with name -{name}- already exists. Would you like to overwrite?")
            override = get_user_decision()
            if override == True:
                print("Material interface overwritten")
                self.materialInterfacesTable[name] = MaterialInterface(matSection1 = materialSection1, matSection2 = materialSection2, interfaceMaterial = material, matIntName = name)
        else:
            self.materialInterfacesTable[name] = MaterialInterface(matSection1 = materialSection1, matSection2 = materialSection2, interfaceMaterial = material, matIntName = name)

class MaterialInterface():
    def __init__(self, matSection1: MaterialSection, matSection2: MaterialSection, interfaceMaterial: Material,matIntName = "MaterialInterface" ) -> None:
        self.material = interfaceMaterial  
        self.materialSection1 = matSection1
        self.materialSection2 = matSection2
        self.name = matIntName
