from collections.abc import Callable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as Scale
from time import time

def timeit(func: Callable) -> Callable:
	"""Prints the time a function took to execute"""
	# This is my favorite functions and if you look at my other repositories on github it becomes quite obvious
	def wrapper(*args, hush: bool = False, **kwargs):
		start= time()
		result = func(*args, **kwargs)
		end = time()
		elapsed = end - start
		if not hush:
			print(f"'{func.__name__}' executed in {elapsed:.4f} seconds")
		return result
	return wrapper

class Utils:
    """Encapsulates of of the utility functions into one object"""
    def __init__(self, path: os.PathLike | str) -> None:
        assert isinstance(path, os.PathLike | str), 'The path should be interpretable for saving images'
        self.path = path
        os.makedirs(self.path, exist_ok = True)

    @timeit
    def findMissing(self, dataframe: pd.DataFrame) -> int:
        """This function prints output if a dataframe has at least one missing value"""
        # This variable holds the final output
        output: int = 0
        mask = dataframe.isna().sum()
        for entry in mask:
            if entry == True:
                output: int = 1
        if output == 1:
            print(f"Identified missing data")
        return output

    @timeit
    def findDuplicates(self, dataframe: pd.DataFrame) -> int:
        """Parses the dataframe and finds if there are any duplicate entries"""
        counter: int = 0
        # I was contemplating how to do this, if two columns have the same feature then their correlation coefficient will be 1
        table: pd.DataFrame = dataframe.corr()
        dimensions = table.shape
        for row in range(dimensions[0]-1):
            for col in range(dimensions[1]-1):
                if row != col and table.iloc[row, col] == 1:
                    print('Duplicate values exist in the data')
                    counter +=1
        return counter

    @timeit
    def meadianSubstitution(self, data: pd.DataFrame) -> pd.DataFrame:
        """Substitues missing values with the median of the feature"""
        # This used to have a -1 for the len() of each loop but apparently it would skip the last row
        output: pd.DataFrame = data.copy()
        for col in range(len(data.columns)):
            series: pd.Series = data.iloc[:, col]
            substitution: pd.Series = series.copy()
            median = series.median() # Should be a float forget what the lsp says
            for entry in range(len(series)):
                if pd.isna(series.iloc[entry]):
                    substitution.iloc[entry] = median
            output.iloc[:, col] = substitution
        return output

    @timeit
    def copyPrune(self, data: pd.DataFrame) -> pd.DataFrame:
        """Removes one of the two features that are the same"""
        output: pd.DataFrame= data.copy()
        return output

    @timeit
    def interpolate(self, dataframe: pd.DataFrame, col: int, categories: list) -> pd.DataFrame:
        """This function takes a dataframe and converts a field to a binary valued 0, 1, 2 ... field"""
        output: pd.DataFrame = dataframe.copy()
        # Isolating our column
        target = output.iloc[:, col].astype("category")
        # Casting to categorical as in the documentation https://pandas.pydata.org/docs/user_guide/10min.html
        target = target.cat.rename_categories(categories) 
        # Re-assigning to dataframe
        output.iloc[:, col] = target
        return output

    @timeit
    def upperTriangle(self, matrix: np.ndarray) -> tuple[list, list]:
        """This function is to return the two lists of indecies to use to plot a matrix"""
        # I need two arrays, one for redundant values and one for non redundant ones
        # Since i need something like
        # 1 - 1
        # 2 - 1
        # 2 - 2
        # 3 - 1
        # 3 - 2
        # ...
        width: int = len(matrix) - 1
        height: int = width
        volume: float = int((width) * (height) / 2) # of a triangle (technically an area)
        redundant: list = [0] * volume
        serial: list = [0] * volume
        holder: int = 0
        for i in range(1, len(matrix) - 1): # This one starts at 1 since the 0 index is on the diagonal I guess we could leave it and it would skip it
            for j in range(0, len(matrix) - 1):
                if i > j:
                    holder += 1
                    redundant[holder] = i
                    serial[holder] = j
                else:
                    continue
        return redundant, serial

    @timeit
    def correlation(self, dataframe: pd.DataFrame, **kwargs) -> None:
        """This function will take the correlation matrix of the dataframe and make a scatter plot for it (col 1 index, col 2 index, corr)"""
        table: pd.DataFrame = dataframe.corr()
        # We only need everything above the diagonal
        rows, cols = self.upperTriangle(table.values)  # Originally solved inside this function now moved to the one above
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        # I'm going to multiply the corr() output by alot since I want to have the delta z visible on the plot outside of just the color heatmap
        ax.scatter(rows, cols, 1000 * table.values[rows, cols], label = 'Pearson corr', c = table.values[rows, cols], cmap = 'YlOrBr')
        plt.title(f"Correlations of the dataframe")
        ax.set_xlabel(f"Col 1")
        ax.set_ylabel(f"Col 2")
        ax.set_zlabel("Pearson correlation of dataframe")
        plt.grid()
        plt.show()
        fig.savefig(self.path + '/Correlations of the dataframe.png')

    @timeit
    def twoCol(self, dataframe: pd.DataFrame, anchor: int) -> None:
        """this function does the 2d scatter plots of the anchor column with all the rest"""
        for index in range(len(dataframe.columns)):
            plt.scatter(dataframe.iloc[:, anchor], dataframe.iloc[:, index], label = f'col {anchor}, {index}')
            plt.xlabel(f'anchor: {dataframe.columns[anchor]}')
            plt.ylabel(f'col {index}: {dataframe.columns[index]}')
            plt.grid()
            plt.title(f'Scatter plot of columns {anchor} {dataframe.columns[anchor]}, {index} {dataframe.columns[index]}')
            plt.show()

    @timeit
    def zScaler(self, dataframe: pd.DataFrame, start: int, stop: int) -> tuple[pd.DataFrame, Scale]:
        """Scales the columns from start to end of the dataframe using z-score scaling"""
        target: pd.DataFrame | pd.Series = dataframe.iloc[:, start:stop]
        transform: Scale = Scale()
        transform.fit(target)
        holder = transform.transform(target)
        output: pd.DataFrame = dataframe.copy()
        output.iloc[:, start:stop] = holder
        return output, transform

    @timeit
    def applyScaler(self, dataframe: pd.DataFrame, start: int, stop: int, scaler: Scale) -> pd.DataFrame:
        """Applies a scaler to columns from start to end of the dataframe"""
        target: pd.DataFrame | pd.Series = dataframe.iloc[:, start:stop]
        holder = scaler.transform(target)
        output: pd.DataFrame = dataframe.copy()
        output.iloc[:, start:stop] = holder
        return output

    @timeit
    def searchPca(self, data: pd.DataFrame) -> None:
        """Calculates and plots the explain values for each feature"""
        # I notmalize the blue curve to the red one so that both are visible in detail
        pca = PCA()
        pca.fit(data)
        fig, ax = plt.subplots()
        ax.plot(range(1, len(data.columns) + 1), pca.explained_variance_ratio_, color = 'red', label = 'Raw variance')
        ax.plot(range(1, len(data.columns) + 1), [sum(pca.explained_variance_ratio_[0:i]) for i in range(len(data.columns))], color = 'blue', label = 'Cummulative variance')
        ax.grid()
        ax.legend()
        ax.set_title('Explain (%) of principal components')
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance')
        plt.show()
        fig.savefig(f'{self.path}/Explain of all principaled components.png')

    def transformPca(self, dataframe: pd.DataFrame, components: int) -> tuple:
        """Applies a PCA to the dataframe"""
        data = dataframe.iloc[:, 2:].values
        obj = PCA(n_components = components)
        fit = obj.fit_transform(data)
        pcaDf = pd.DataFrame(fit, columns = [f'P.C. {i+1}' for i in range(components)])
        return pcaDf, obj.explained_variance_ratio_

    @timeit
    def componentSearch(self, dataframe: pd.DataFrame, cutoff: float | int) -> int:
        """
        This function will do search on the results of the pca for different components until it find the minimum number of omponents with a cutoff explain
        Will return the components to keep
        """
        for n in range(len(dataframe.columns) - 2): # The -2 is because we skip the first 2 columns
            dfPca, explain = self.transformPca(dataframe, n)
            if sum(explain) >= cutoff:
                output: int = n
                break
        if output:
            return output
        else:
            return -1

    @timeit
    def implementPca(self, dataframe: pd.DataFrame, cutoff: float | int, verbose: bool = False) -> tuple:
        """
        This function will return the pca resutls for the first Pca run that passes cutoff
        """
        components = self.componentSearch(dataframe, cutoff)
        pca = self.transformPca(dataframe, components)
        if verbose:
            print(f'Total explain: {sum(pca[1])}\nComponents {components}')
        return pca


