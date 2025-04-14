from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
import xgboost as xgb
from .preprocess import DataFrame
from pydantic import BaseModel
import pandas as pd


class hyperparameters(BaseModel):
    train_strategy: str="train_test_split"
    null_values_treatment: list[str]="drop"
    test_size: float=0.2
    cv: int =None 
    random_state: int=42

class experiment:
    def __init__(self, model, dataset: DataFrame, hyperparameters: hyperparameters, metrics:list[str]):
        self.model = model
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.train_strategy = hyperparameters.train_strategy
        self.dataframes = [dataset.null_values_treatment(treatment) for treatment in hyperparameters.null_values_treatment]
        self.columns = ["model", "null_values_treatment", "test_size", "cv", "random_state"] + metrics
        self.results = pd.DataFrame(columns=self.columns)
    def run(self):
        return None 

class project:
    def __init__(self, name: str, description: str, dataset: DataFrame, metrics:list[str]):
        self.name = name
        self.description = description
        self.dataset = dataset
        self.experiments = []

    def add_experiment(self, experiment):
        self.experiments.append(experiment)

    def execute_experiments(self):
        results = pd.DataFrame(columns=set([experiment.columns for experiment in self.experiments]))

    def get_experiment_results(self):
        return self.experiments
    

classifiers = {
    'Random': DummyClassifier(strategy='uniform', random_state=42),
    'XGBoost-1': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    'XGBoost-2': xgb.XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=5, random_state=42),
    'Random Forest-1': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Random Forest-2': RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, random_state=42),
    'SVM-1': SVC(kernel='rbf', C=1.0, random_state=42),
    'SVM-2': SVC(kernel='linear', C=0.1, random_state=42),
    'Gradient Boosting-1': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'Gradient Boosting-2': GradientBoostingClassifier(n_estimators=200, learning_rate=0.01, max_depth=5, random_state=42),
    'KNN-1': KNeighborsClassifier(n_neighbors=5),
    'KNN-2': KNeighborsClassifier(n_neighbors=10, weights='distance'),
    'Neural Network-1': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    'Neural Network-2': MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42),
    'Logistic Regression-1': LogisticRegression(C=1.0, random_state=42),
    'Logistic Regression-2': LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    'AdaBoost-1': AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42),
    'AdaBoost-2': AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
}
