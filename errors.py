from typing import List, Tuple
class Errors():
    
    def __init__(self, error_and_traceback : List[Tuple[Exception, str]]):
        self.error_and_traceback = error_and_traceback
    
    def __str__(self):
        result = ""
        for i, (error, traceback) in enumerate(self.error_and_traceback):
            result += f"Try {i+1}: \n"
            result += f"Error : {error}\n"
            result += f"Traceback : {traceback}\n"
        return result
    
    def __dict__(self):
        return [
            {
            "try" : i,
            "error" : str(error),
            "traceback" : str(traceback)
        }for i, (error, traceback) in enumerate(self.error_and_traceback)
        ]
    
    def __iter__(self):
        return iter(self.error_and_traceback)
    
    def __len__(self):
        return len(self.error_and_traceback)
    
    def append(self, error_and_traceback : Tuple[Exception, str]):
        self.error_and_traceback.append(error_and_traceback)

