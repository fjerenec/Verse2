from data import get_user_decision

class Materials():
    def __init__(self) -> None:
        self.materialsTable = {}
        
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
            else:
                return
        else:
            if isIsotropic:
                self.materialsTable[materialName] = Isotropic(materialName=materialName)
            else:
                self.materialsTable[materialName] = Anisotropic(materialName=materialName)

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

    def get_material_vars(self):
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


class MaterialSection():

    MaterialSections = {}

    def __init__(self) -> None:
        pass

class MaterialInterface():

    MaterialInterfaces = {}        

    def create_material_interface(self,  interfaceMaterial: Material, matSection1: MaterialSection, matSection2: MaterialSection,matIntName = "MaterialInterface" ):
        self.interfaceMat = interfaceMaterial
        self.matSec1 = matSection1
        self.matSec2 = matSection2
        MaterialInterface.MaterialInterfaces[matIntName] = self

    def find_interface_bonds(self):
        pass