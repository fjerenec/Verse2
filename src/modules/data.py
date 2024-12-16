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
    
    def get_number_of_points(self):
        return self.IDarray.shape[0]

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



class State():
    """
    This type stores the state (so the results) of a increment or step of the simulation. It does not have fixed result categories.
    Each result is stored in a dictionary with the name of the result as the key (e.g. "stress") and the result as the value.
    """
    def __init__(self) -> None:
        self.state_data = {}

    def add_state_data(self, name: str, data):
        self.state_data[name] = data

    def is_from_step(self):
        self.from_step = True

    def is_from_increment(self):
        self.from_increment = True

    def has_state_data(self, name: str) -> bool:
        """
        Check if the given name exists as a key in the `state_data` dictionary.

        Parameters:
            name (str): The name to check for in the `state_data` dictionary.

        Returns:
            bool: True if the name exists as a key in the `state_data` dictionary, False otherwise.
        """
        if name in self.state_data.keys():
            return True
        else: return False

class HistoryOutput():
    """
    Simply stores "State" objects into a list.
    Each index of the list corresponds to a step in the simulation.
    """
    def __init__(self) -> None:
        self.history = []
    
    def add_state_to_history(self, state: State):
        self.history.append(state)