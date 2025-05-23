# As is usual in alphabetical order
from collections.abc import Callable
import joblib
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import numpy as np
import optuna
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.svm import SVC
from tqdm import tqdm
from typing import Sequence
from typing import Any
from utils import Utils
from utils import timeit

class RNCV:
    """This class implements the main required class of the assignment being the repeated cross validation"""
    def __init__(self,
                 name: str,
                 data: pd.DataFrame,
                 path: str,
                 estimators: list,
                 metrics: list,
                 loops: int,
                 innerFolds: int,
                 outerFolds: int,
                 *args,
                 hyperparameters: list | None = None,
                 **kwargs) -> None:
        # I usually go for isinstance in a if statement but I saw this recently and liked how many fewer lines it is
        assert isinstance(name, str) == True, f'RNCV requires that name is a string, expected str got {type(name)} instead.\n'
        assert isinstance(data, pd.DataFrame) == True, f'RNCV requires that data is a pandas dataframe, expected pd.DataFrame got {type(data)} instead.\n'
        assert isinstance(path, str) == True, f'RNCV requires that path is a path like object, expected str got {type(path)} instead.\n'
        assert isinstance(estimators, list) == True, f'RNCV requires that estimators is a list with sklearn classes, expected list got {type(estimators)} instead.\n'
        assert isinstance(metrics, list) == True, f'RNCV requires that metrics is a list with sklearn classes, expected list got {type(metrics)} instead.\n'
        assert isinstance(loops, int) == True, f'RNCV requires that loops is a positive non 0 integer, expected int got {type(loops)} instead.\n'
        assert isinstance(innerFolds, int) == True, f'RNCV requires that innerFolds is a positive non 0 integer, expected int got {type(innerFolds)} instead.\n'
        assert isinstance(outerFolds, int) == True, f'RNCV requires that outerFolds is a positive non 0 integer, expected int got {type(outerFolds)} instead.\n'
        assert isinstance(hyperparameters, list | None) == True, f'RNCV requires that hyperparameters is a list of different key value pairs for each method, expected list got {type(hyperparameters)} instead.\n'
        self.name: str = name
        self.figures = f'../figures/RNCV/{self.name}/' # For saving only the figures
        self.data: pd.DataFrame = data
        self.dataCopy: pd.DataFrame = data
        self.estimators: list = estimators
        self.metrics = metrics
        self.loops: int = loops
        self.iF: int = innerFolds
        self.oF: int = outerFolds
        self.utils = Utils(path = self.figures)
        self.path: str = path # For saving the model
        self.hyper: list | None = hyperparameters # for method arguments
        self.args: tuple = args # for metrics
        self.kwargs: dict = kwargs 
        self.shape = (self.loops, self.oF, self.iF)
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.figures, exist_ok=True)
        # pre allocating memory
        # In every loop we write over these lists and the final list should have all the values from all the respective folds
        # In the next loop the lists will be overwritten
        self.outerFoldModels: list = [0] * self.oF
        self.outerFoldPredictions: list = [0] * self.oF
        self.outerFoldEvaluations: np.ndarray = np.array(
            [
                [
                    [0] * len(self.estimators)
                ] * len(self.metrics)
            ] * self.oF, dtype = np.float64
        )
        self.innerFoldModels: list = [0] * self.iF
        self.innerFoldPredictions: list = [0] * self.iF
        self.innerFoldEvaluations: np.ndarray = np.array(
            [
                [
                    [0] * len(self.estimators)
                ] * len(self.metrics)
            ] * self.iF, dtype = np.float64
        )
        # Now I will make lists to append these to so that I can keep each loop
        self.wrapperOFM: list = [0] * self.loops
        self.wrapperOFP: list = [0] * self.loops
        self.wrapperOFE: np.ndarray = np.array(
            [
                [
                    [
                        [0] * len(self.estimators)
                    ] * len(self.metrics)
                ] * self.oF, 
            ] * self.loops,
            dtype = np.float64
        )
        self.wrapperIFM: list = [0] * self.loops
        self.wrapperIFP: list = [0] * self.loops
        self.wrapperIFE: np.ndarray = np.array(
            [
                [
                    [
                        [0] * len(self.estimators)
                    ] * len(self.metrics)
                ] * self.iF,
            ] * self.loops,
            dtype = np.float64
        )
        # Look at the fit and loop functions for what happens with these lists

    def __len__(self) -> int:
        """Returns the number of iterations that will be done"""
        return self.loops * self.oF * self.iF

    @timeit
    def fit(self,
            fs: int | None = None,
            optimize: bool = False,
            innerPlots: bool = True,
            printEvals: bool = True,
            fsMethod: Callable | None = None) -> None:
        """Splits the dataset and trains - evaluates all models with the given splits"""
        if fsMethod is None:
            fsMethod = self.featureSelection
        self.data = self.preprocess(self.data)
        if fs is not None:
            holder = self.data['diagnosis']
            self.data = fsMethod(self.data, n = fs, holder = holder.to_numpy(int))
            self.data['diagnosis'] = holder
            del holder
        for loop in tqdm(range(self.loops)):
            train, trainLabels, self.test, self.testLabels = self._split(
                self.data,
                splits = self.oF,
                hush = True
            )
            for split in range(self.oF):
                # Since my split function outputs numpy arrays I need to somehow get the label back in for the next split(call)
                holder = pd.DataFrame(train[split])
                holder['diagnosis'] = pd.Series(trainLabels[split, :])
                self.train, self.trainLabels, self.val, self.valLabels = self._split(
                    holder,
                    splits = self.iF,
                    hush = True
                )
                del holder
                for index in range(self.iF):
                    if optimize:
                        self.optimizeAll(
                            fbeta_score,
                            self.train,
                            self.trainLabels,
                            self.val,
                            self.valLabels,
                            index = index
                        )
                    # Train
                    self.innerFoldModels[index] = self.fitting(
                        self.estimators,
                        self.train[index],
                        self.trainLabels[index],
                        arguments = self.hyper,
                        hush = True
                    )
                    # Predict
                    self.innerFoldPredictions[index] = self.useFitModels(
                        self.innerFoldModels[index],
                        self.val[index],
                        hush = True
                    )
                    # Evaluate
                    self.innerFoldEvaluations[index] = self.applyMetrics(
                        self.metrics,
                        self.innerFoldModels[index],
                        self.valLabels[index],
                        self.innerFoldPredictions[index],
                        arguments = self.args,
                        hush = True
                    )
                self.wrapperIFM[loop] = self.innerFoldModels
                self.wrapperIFP[loop] = self.innerFoldPredictions
                self.wrapperIFE[loop] = self.innerFoldEvaluations
                if printEvals:
                    print(self.innerFoldEvaluations)
                if innerPlots:
                    self.plotKfoldInner_(
                        self.innerFoldModels,
                        self.metrics,
                        self.innerFoldEvaluations,
                        name = f'Inner fold boxplots ({loop}, {split})',
                        hush = True
                    )
                if optimize:
                    self.optimizeAll(
                        fbeta_score,
                        train,
                        trainLabels,
                        self.test,
                        self.testLabels,
                        index = split
                    )
                # Train
                self.outerFoldModels[split] = self.fitting(
                    self.estimators,
                    train[split],
                    trainLabels[split],
                    arguments = self.hyper,
                    hush = True
                )
                # Predict
                self.outerFoldPredictions[split] = self.useFitModels(
                    self.outerFoldModels[split],
                    self.test[split],
                    hush = True
                )
                # Evaluate
                self.outerFoldEvaluations[split] = self.applyMetrics(
                    self.metrics,
                    self.outerFoldModels[split],
                    self.testLabels[split],
                    self.outerFoldPredictions[split],
                    arguments = self.args,
                    hush = True
                )
            self.wrapperOFM[loop] = self.outerFoldModels
            self.wrapperOFP[loop] = self.outerFoldPredictions
            self.wrapperOFE[loop] = self.outerFoldEvaluations
            if printEvals:
                print(self.outerFoldEvaluations)
            if innerPlots:
                self.plotKfoldInner_(
                    self.outerFoldModels,
                    self.metrics,
                    self.outerFoldEvaluations,
                    name = f'Outer fold boxplots ({loop})',
                    hush = True
                )
        # Plotting
        self.plotKfoldSummary(
            self.outerFoldModels,
            self.metrics,
            self.wrapperOFE,
            name = f'All loop outer folds boxplots',
            hush = True
        )
        self.plotKfoldSummary(
            self.innerFoldModels,
            self.metrics,
            self.wrapperIFE,
            name = f'All loop inner fold boxplots',
            hush = True
        )

    @timeit
    def bonusFS(self, data: pd.DataFrame, n: int, holder: Any) -> None:
        """Does a select k best feature selection based on the chi^2"""
        return pd.DataFrame(SelectKBest(chi2, k=n).fit_transform(data, holder))

    @timeit
    def bonusFS2(self, data: pd.DataFrame, n: int, holder: Any) -> None:
        """Does a select k best feature selection based on the mutual information"""
        return pd.DataFrame(SelectKBest(mutual_info_classif, k=n).fit_transform(data, holder))

    @timeit
    def optimizeAll(self, score: Callable, tr: list | np.ndarray, trl: list | np.ndarray, vl: list | np.ndarray, vll: list | np.ndarray, index: int) -> None:
        """Updates the self.hyper parameter"""
        self.hyper = [
            self.optimizeLR(
                score,
                tr[index],
                trl[index],
                vl[index],
                vll[index],
            ),
            self.optimizeGNB(
                score,
                tr[index],
                trl[index],
                vl[index],
                vll[index],
            ),
            self.optimizeLDA(
                score,
                tr[index],
                trl[index],
                vl[index],
                vll[index],
            ),
            self.optimizeSVC(
                score,
                tr[index],
                trl[index],
                vl[index],
                vll[index],
            ),
            self.optimizeRFC(
                score,
                tr[index],
                trl[index],
                vl[index],
                vll[index],
            ),
            self.optimizeLGBM(
                score,
                tr[index],
                trl[index],
                vl[index],
                vll[index],
            )
    ]

    def findBest(self, metric: Callable = fbeta_score) -> Callable:
        """Will search the lists or the models with the best evaluation metric"""
        winner: list = [0, 0] # This is to keep the index
        currentBest = 0
        # We neet to iterate self.wrapperOFM and pick the model with the best value in the same index in wrapperOFE
        # wraperOFM is loops x oF x methods
        # WrapperOFE is loops x oF x metrics x methods
        # We will always look for logistic regression so that means the last index of wrapper ofm is alwasy 0
        indexer = self.wrapperOFE.shape
        for outer in range(indexer[0]):
            for inner in range(indexer[1]):
                # This is where the two lists can't be indexed the smae anymore
                # Check best value
                if self.wrapperOFE[outer, inner, 3, 0] >= currentBest:
                    currentBest = self.wrapperOFE[outer, inner, 3, 0]
                    currentWinner = [outer, inner]
        winner = currentWinner
        best = self.wrapperOFM[winner[0]][winner[1]][0]
        return best

    def reTrainBest(self, best: Callable, data: pd.DataFrame) -> Callable:
        """Takes the best model and re trains it with the whole data"""
        # split data
        data = self.preprocess(data)
        train, labels = self.isolator(data)
        # fit
        best.fit(train.to_numpy(dtype = np.float64), labels.to_numpy(dtype = int))
        return best

    @timeit
    def winner(self) -> None:
        """Saves the winner with the whole data"""
        self.bestModel = self.reTrainBest(self.findBest(metric = fbeta_score), self.dataCopy)
        # save model
        joblib.dump(self, '../models/winner.pkl')

    @timeit
    def predict(self, data: pd.DataFrame) -> None:
        """Given data produces their labels, will perform the median substitution"""
        data = self.preprocess(data)
        predictions = self.bestModel.predict(data)
        return predictions

    @timeit
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies the utils class pipeline to the data"""
        data = self.utils.interpolate(data, 1, [0, 1]) # makes benign 0 and malignant 1
        if self.utils.findMissing(data):
            data = self.utils.meadianSubstitution(data)
        if self.utils.findDuplicates(data):
            data = self.utils.copyPrune(data)
        return data

    @timeit
    def featureSelection(self, data: pd.DataFrame, n: int | float, holder: Any) -> pd.DataFrame:
        """Default feature selection (pca)"""
        self.pca = self.fitPca(data = data, n = n)
        return self.applyPca(data = data, pca = self.pca)

    @timeit
    def fitPca(self, data: pd.DataFrame, n: int | float) -> PCA:
        """Fits the pca and returns the object"""
        return PCA(n_components = n).fit(data)

    @timeit
    def applyPca(self, data: pd.DataFrame, pca: PCA) -> pd.DataFrame:
        """Returns the dataframe with the transformation to n components"""
        return pd.DataFrame(pca.fit_transform(data))

    @timeit
    def _split(self,
               data: pd.DataFrame,
               splits: int) -> tuple:
        """Splits the data given into as many pieces as the splits argument is"""
        # There must be some way to make this blow up with non divisable values
        # I want an array for both the majority holder (train set) and the minority holder (train or val) set
        # I'm just going to wrap the array of the slices in a one dimensional higher array, and each index of the third dimension in each split
        # This one should have 1/splits rows
        piece, label = self.isolator(data, hush = True)
        holder = piece.values.shape
        rows = holder[0] # 512
        cols = holder[1] # 31
        del holder
        wedges: np.ndarray = np.array(
            [
                [
                    [0] * cols 
                ] * int(rows/splits)
            ] * splits
        , dtype = np.float64)
        # Labels have 1 column
        wedgeLabels: np.ndarray = np.array(
            [
                [0] * int(rows/splits)
            ] * splits
        , dtype = np.float64)
        # This one should have however many rows are left over in the data from the previous array
        reciprocal: np.ndarray = np.array(
            [
                [
                    [0] * cols
                ] * int(rows * ((splits - 1)/splits))
            ] * splits
        , dtype = np.float64)
        # Labels have 1 column
        reciprocalLabels: np.ndarray = np.array(
            [
                [0] * int(rows * ((splits - 1)/splits))
            ] * splits
        , dtype = np.float64)
        for split in range(splits):
            # pieces
            # Isolating the reciprocal of the 1/splits wedge of the data
            restLeft: np.ndarray = piece.iloc[0:int(rows/splits) * split, :].values # From the begining up till the wedge
            restRight: np.ndarray = piece.iloc[int(rows/splits) * (split + 1):-1, :].values # From the wedge end till the end
            # Isolating the 1/splits wedge in the data
            wedges[split, :, :] = piece.iloc[int(rows/splits) * split:int(rows/splits) * (split + 1), :].values # Just the wedge
            # Assigning the reciprocals
            reciprocal[split, :len(restLeft), :] = restLeft
            reciprocal[split, len(restLeft):, :] = restRight
            # labels
            # This is the same as before with one less dimension for the labels
            restLeftLabel: np.ndarray = label.iloc[0:int(rows/splits) * split].values
            restRightLabel: np.ndarray = label.iloc[int(rows/splits) * (split + 1):-1].values
            # Allocating indexes to memory
            wedgeLabels[split, :] = label.iloc[int(rows/splits) * split:int(rows/splits) * (split + 1)].values
            reciprocalLabels[split, :len(restLeftLabel)] = restLeftLabel
            reciprocalLabels[split, len(restLeftLabel):] = restRightLabel
        return reciprocal, reciprocalLabels, wedges, wedgeLabels

    @timeit
    def classifications(self,
                        methods: Sequence[Callable],
                        train: pd.DataFrame,
                        pred: pd.DataFrame,
                        val: pd.DataFrame,
                        arguments: list | None = None) -> np.ndarray:
        """Calls the method on the data given and returns outputs"""
        # for no re computations
        scale = len(methods)
        # Variable to keep outputs
        predictions = np.array(
            [
                np.zeros(
                    val.shape[0]
                )
            ] * scale
        , dtype = np.float64)
        # I think it always has arguments but just in case
        if arguments:
            index = 0
            for model, args in zip(methods, arguments):
                object = model(**args)
                object.fit(train, pred)
                predictions[index] = object.predict(val)
                index += 1
        else:
            for index, model in enumerate(methods):
                object = model()
                object.fit(train, pred)
                predictions[index] = object.predict(val)
        return predictions

    @timeit
    def plotKfoldInner_(self, methods: list , metrics: list, scores: np.ndarray, name: str, showFig: bool = True) -> None:
        """Plots the output of fit"""
        holder = len(methods[0])
        temp = len(metrics)
        fig, ax = plt.subplots(nrows=holder, ncols=temp, figsize=(6*holder, 4*temp), sharey = True)
        for i in range(temp): # Should iterate input col
            for index in range(holder): # Should iterate input row
                ax[index, i].boxplot(scores[:, i, index], showmeans=True, meanline=True, sym = '.') # index, i is row, col in matplotlib
                ax[index, i].set_title(f"Metric: {metrics[i].__name__}") # Funciton are first class objects in python so __name__ just returns the function name string
                ax[index, i].grid()
                ax[index, i].set_ylabel(f'{metrics[i].__name__} Value')
                ax[index, i].set_xlabel(f'Method: {methods[0][index].__class__.__name__}')
        if showFig:
            plt.show()
        fig.savefig(self.figures + name + '.png')

    @timeit
    def plotKfoldSummary(self, methods: list , metrics: list, scores: np.ndarray, name: str, showFig: bool = True) -> None:
        """Plots the output of fit"""
        holder = len(methods[0])
        temp = len(metrics)
        fig, ax = plt.subplots(nrows=holder, ncols=temp, figsize=(6*holder, 4*temp), sharey = True)
        for i in range(temp): # Should iterate input col
            for index in range(holder): # Should iterate input row
                ax[index, i].boxplot(scores[:, :, i, index].flatten(), showmeans=True, meanline=True, sym = '.') # index, i is row, col in matplotlib
                ax[index, i].set_title(f"Metric: {metrics[i].__name__}") # Funciton are first class objects in python so __name__ just returns the function name string
                ax[index, i].grid()
                ax[index, i].set_ylabel(f'{metrics[i].__name__} Value')
                ax[index, i].set_xlabel(f'Method: {methods[0][index].__class__.__name__}')
        if showFig:
            plt.show()
        fig.savefig(self.figures + name + '.png')

    @timeit
    def applyMetrics(self,
                     metrics: Sequence[Callable],
                     options: Sequence[Callable],
                     truth: np.ndarray,
                     data: np.ndarray,
                     arguments: list | None = None) -> np.ndarray:
        """Applies all the given metrics to the data assuming as ground truth the truth given"""
        # To not re compute this
        holder = len(options)
        # Variable to keep the metrics
        output = np.array(
            [
                [
                    0
                ] * holder
            ] * len(metrics)
        , dtype = np.float64)
        # So we will end up with the following
        # [[[option1], [option2], [option3], ...], # metric1
        # [[option1], [option2], [option3], ...], # metric2
        # [...], # ...3
        # [...], # ...4
        # ...]
        if arguments:
            i = 0
            for metric, args in zip(metrics , arguments):
                for index in range(holder):
                    output[i, index] = metric(truth, data[index], **args)
                i += 1
            del i
        else:
            for i, metric in enumerate(metrics):
                for index in range(holder):
                    output[i, index] = metric(truth, data[index])
        return output

    @timeit
    def visualiseEvaluations(self,
                             input: np.ndarray,
                             methods: Sequence[Callable],
                             metrics: Sequence[Callable],
                             name1: str,
                             name2: str,
                             saveOnly: bool = False,
                             verbose: bool = False) -> None:
        """Takes the output of the applyMetrics function and displayes it with matplotlib"""
        holder: int = len(methods)
        temp: int = len(metrics)
        fig, ax = plt.subplots(nrows=temp, ncols=1, figsize=(7, 8*holder))
        for i in range(temp): # Should iterate input col
            title_: str = metrics[i].__name__
            for index in range(holder): # Should iterate input row
                lab: str = methods[index].__class__.__name__
                ax[i].scatter(index, input[i, index], label = f'{lab}')
            ax[i].set_title(f"Metric: {title_}") # Funciton are first class objects in python so __name__ just returns the function name string
            ax[i].set_ylabel(f'{title_} Value')
            ax[i].set_xlabel('Method')
            ax[i].legend()
            ax[i].grid()
        if not saveOnly:
            plt.show()
        fig.savefig(self.figures + f'Evaluations loop inner fold {name1} outer fold {name2}.png')
        if verbose:
            print(input)

    @timeit
    def fitting(self,
                methods: Sequence[Callable],
                train: pd.DataFrame,
                pred: pd.DataFrame,
                arguments: list | None = None) -> list:
        """The purpose of this function is to return the model objects form sci kit learn after the fit happens"""
        # I made this function so that I don't re train the models for each loop of the bootstrap
        output = [0] * len(methods)
        if arguments:
            index = 0
            for model, args in zip(methods, arguments):
                object = model(**args)
                object.fit(train,pred)
                output[index] = object
                index += 1
            del index
        else:
            for index, model in enumerate(methods):
                object = model()
                object.fit(train, pred)
                output[index] = object
        return output

    @timeit
    def useFitModels(self,
                     fitMethods: Sequence[Any],
                     val: pd.DataFrame) -> np.ndarray:
        """This function pairs with fitting to not have to call the .fit() for each iteration of a bootstrap"""
        # The type hint needs any since callable is a function and the .predict is not a function method
        scale = len(fitMethods)
        # Check the regression funtion for why this data structure
        predictions = np.array(
            [
                np.zeros(
                    val.shape[0]
                )
            ] * scale
        , dtype = np.float64)
        for index, model in enumerate(fitMethods):
            predictions[index] = model.predict(val)
        return predictions

    @timeit
    def isolator(self,
                 data: pd.DataFrame,
                 keyword: str = 'diagnosis') -> tuple:
        """Isolates the x's and y's of the dataframe into different variables"""
        x = data.drop(columns = [keyword]) # everything other than the keyword
        y = data[keyword]
        return x, y

    @timeit
    def save_(self, filename: str | None = None) -> None:
        """Uses joblib to save this class as a pickle file in the self.path directory"""
        if filename is None:
            filename = self.name
        joblib.dump(self, self.path + filename + '.pkl')

    @timeit
    def optimizeLR(self,
                   metric: Callable,
                   train: pd.DataFrame,
                   preds: pd.DataFrame,
                   val: pd.DataFrame,
                   truth: pd.DataFrame,
                   trials: int = 100) -> dict:
        """Returns LogisticRegression class hyperparameters after tuning"""
        def objective(trial: optuna.trial.Trial,
                      metric: Callable = metric,
                      train: pd.DataFrame = train,
                      preds: pd.DataFrame = preds,
                      val: pd.DataFrame = val,
                      truth: pd.DataFrame = truth) -> float:
            """Defines an objective to optimize"""
            c = trial.suggest_float('C', 0.1, 10.0)
            max_i = trial.suggest_int('max_iter', 50, 500)
            intercept = trial.suggest_categorical('fit_intercept', [True, False])
            sol = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
            ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            model = LogisticRegression(
                C = c,
                max_iter = max_i,
                fit_intercept=intercept,
                solver = sol,
                l1_ratio = ratio
            )
            model.fit(train, preds)
            output = model.predict(val)
            return metric(output, truth, beta = 2)
        study = optuna.create_study()
        study.optimize(objective, n_trials = trials, timeout = 500)
        return study.best_params

    @timeit
    def optimizeGNB(self,
                    metric: Callable,
                    train: pd.DataFrame,
                    preds: pd.DataFrame,
                    val: pd.DataFrame,
                    truth: pd.DataFrame,
                    trials: int = 100) -> dict:
        """Returns Naive Bayes Classifier class hyperparameters after tuning"""
        def objective(trial: optuna.trial.Trial,
                      metric: Callable = metric,
                      train: pd.DataFrame = train,
                      preds: pd.DataFrame = preds,
                      val: pd.DataFrame = val,
                      truth: pd.DataFrame = truth) -> float:
            """Defines an objective to optimize"""
            var = trial.suggest_float('var_smoothing', 1e-12, 1e-6)
            model = GaussianNB(var_smoothing = var)
            model.fit(train, preds)
            output = model.predict(val)
            return metric(output, truth, beta = 2)
        study = optuna.create_study()
        study.optimize(objective, n_trials = trials, timeout = 500)
        return study.best_params

    @timeit
    def optimizeLDA(self,
                    metric: Callable,
                    train: pd.DataFrame,
                    preds: pd.DataFrame,
                    val: pd.DataFrame,
                    truth: pd.DataFrame,
                    trials: int = 100) -> dict:
        """Returns LinearDiscriminantAnalysis class hyperparameters after tuning"""
        def objective(trial: optuna.trial.Trial,
                      metric: Callable = metric,
                      train: pd.DataFrame = train,
                      preds: pd.DataFrame = preds,
                      val: pd.DataFrame = val,
                      truth: pd.DataFrame = truth) -> float:
            """Defines an objective to optimize"""
            solver = trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen'])
            model = LinearDiscriminantAnalysis(
                solver = solver,
            )
            model.fit(train, preds)
            output = model.predict(val)
            return metric(output, truth, beta = 2)
        study = optuna.create_study()
        study.optimize(objective, n_trials = trials, timeout = 500)
        return study.best_params

    @timeit
    def optimizeSVC(self,
                    metric: Callable,
                    train: pd.DataFrame,
                    preds: pd.DataFrame,
                    val: pd.DataFrame,
                    truth: pd.DataFrame,
                    trials: int = 100) -> dict:
        """Returns SVR class hyperparameters after tuning"""
        def objective(trial: optuna.trial.Trial,
                      metric: Callable = metric,
                      train: pd.DataFrame = train,
                      preds: pd.DataFrame = preds,
                      val: pd.DataFrame = val,
                      truth: pd.DataFrame = truth) -> float:
            """Defines an objective to optimize"""
            c = trial.suggest_float('C', 1.0, 5.0)
            gamma = trial.suggest_categorical('gamma', ['auto', 'scale'])
            degree = trial.suggest_int('degree', 2, 5)
            coef = trial.suggest_float('coef0', 0.1, 1.0)
            model = SVC(C=c, degree=degree, coef0=coef, gamma=gamma, random_state = 42)
            model.fit(train, preds)
            output = model.predict(val)
            return metric(output, truth, beta = 2)
        study = optuna.create_study()
        study.optimize(objective, n_trials = trials, timeout = 300)
        return study.best_params

    @timeit
    def optimizeRFC(self,
                    metric: Callable,
                    train: pd.DataFrame,
                    preds: pd.DataFrame,
                    val: pd.DataFrame,
                    truth: pd.DataFrame,
                    trials: int = 100) -> dict:
        """Returns Random Forest Clssifier class hyperparameters after tuning"""
        def objective(trial: optuna.trial.Trial,
                      metric: Callable = metric,
                      train: pd.DataFrame = train,
                      preds: pd.DataFrame = preds,
                      val: pd.DataFrame = val,
                      truth: pd.DataFrame = truth) -> float:
            """Defines an objective to optimize"""
            est = trial.suggest_int('n_estimators', 10, 200)
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            model = RandomForestClassifier(
                n_estimators = est,
                criterion = criterion,
            )
            model.fit(train, preds)
            output = model.predict(val)
            return metric(output, truth, beta = 2)
        study = optuna.create_study()
        study.optimize(objective, n_trials = trials, timeout = 500)
        return study.best_params

    @timeit
    def optimizeLGBM(self,
                     metric: Callable,
                     train: pd.DataFrame,
                     preds: pd.DataFrame,
                     val: pd.DataFrame,
                     truth: pd.DataFrame,
                     trials: int = 100) -> dict:
        """Returns light GBM class hyperparameters after tuning"""
        def objective(trial: optuna.trial.Trial,
                      metric: Callable = metric,
                      train: pd.DataFrame = train,
                      preds: pd.DataFrame = preds,
                      val: pd.DataFrame = val,
                      truth: pd.DataFrame = truth) -> float:
            """Defines an objective to optimize"""
            lr = trial.suggest_float('learning_rate', 0.0, 0.5)
            alpha = trial.suggest_float('reg_alpha', 0.0, 1.0)
            lam = trial.suggest_float('reg_lambda', 0.0, 1.0)
            model = LGBMClassifier(
                learning_rate = lr,
                reg_alpha = alpha,
                reg_lambda=lam
            )
            model.fit(train, preds)
            output = model.predict(val)
            return metric(output, truth, beta = 2)
        study = optuna.create_study()
        study.optimize(objective, n_trials = trials, timeout = 500)
        return study.best_params

