import pandas as pd

class RNCV:
    """This class implements the main required class of the assignment being the repeated cross validation

    Arguments:

        """
    def __init__(self,
                 data: pd.DataFrame,
                 estimators: list,
                 loops: int,
                 innerFolds: int,
                 outerFolds: int,
                 *args) -> None:
        # I usually go for isinstance in a if statement but I saw this recently and liked how many fewer lines it is
        assert(isinstance(data, pd.DataFrame) == True, f'')
        assert(isinstance(estimators, list) == True, f'')
        assert(isinstance(loops, int) == True, f'')
        assert(isinstance(innerFolds, int) == True, f'')
        assert(isinstance(outerFolds, int) == True, f'')
        self.data: pd.DataFrame = data
        self.estimators: list = estimators
        self.loops: int = loops
        self.iF: int = innerFolds
        self.oF: int = outerFolds
        self.args = args

    def fit(self, data: pd.DataFrame) -> None:
        """Documentation"""
        # body
        return None

    def predict(self, arg: type) -> None:
        """Documentation"""
        # body
        return None

    def plot(self, arg: type) -> None:
        """Documentation"""
        # body
        return None

