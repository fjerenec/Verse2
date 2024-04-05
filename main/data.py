import numpy as np


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
    """Meant for storing nodes"""
    def __init__(self,arrayOfIDs: np.ndarray[int,1], name: str = "Set") -> None:
        self.name = name
        self.IDarray = arrayOfIDs
        self.IDTable = {key: None for key in arrayOfIDs}

    def is_in_set(self, ID: int) -> bool:
        if ID in self.IDTable.keys():
            return True
        else: return False

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