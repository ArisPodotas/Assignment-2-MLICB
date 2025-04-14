import joblib
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import fbeta_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import sys
sys.path.append('../src/')
from utils import Utils
from classes import RNCV

def main() -> None:
    """Saves winner"""
    model = joblib.load('../models/Optimized/Optimized.pkl')
    model.winner()

if __name__ == "__main__":
    main()


