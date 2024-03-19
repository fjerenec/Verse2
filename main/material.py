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

class Material():

    def __init__(self,materialName:str = "Material") -> None:
        self.name = materialName

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


    
