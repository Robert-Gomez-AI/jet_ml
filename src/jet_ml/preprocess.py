import pandas as pd
from .plots.plot_configs import *
from sklearn.model_selection import train_test_split
from .plots.plots import *



class DataFrame(pd.DataFrame):
    """
    A class that extends pandas.DataFrame with additional methods for data preprocessing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = self.shape[0]
        self.p = self.shape[1]

    def null_values_treatment(self, method: str, column: list[str] = None):
        """
        Treats null values in the DataFrame based on the specified method.

        Args:
            method (str): The method to use for treating null values.
            - "drop": Drops rows with any null values.
            - "mean": Replaces null values with the mean of the column.
            - "median": Replaces null values with the median of the column.
            - "mode": Replaces null values with the mode of the column.

        Returns:
            DataFrame: Returns self after treating null values.
        """
        if column is None:
            try:
                if method == "drop":
                    self._update_inplace(self.drop(self[self.isna().any(axis=1)].index))
                elif method == "mean":
                    self._update_inplace(self.fillna(self.mean()))
                elif method == "median":
                    self._update_inplace(self.fillna(self.median()))
                elif method == "mode":
                    self._update_inplace(self.fillna(self.mode().iloc[0]))
                    print(self)
                    print(self.isna().sum())
                return self
            except Exception as e:
                print(f"Error: {e}")
                return self
        else:
            for col in column:
                if method == "drop":
                    self._update_inplace(self.drop(self[self[col].isna()].index))
                elif method == "mean":
                    self._update_inplace(self.fillna(self[col].mean()))
                elif method == "median":
                    self._update_inplace(self.fillna(self[col].median()))
                elif method == "mode":
                    self._update_inplace(self.fillna(self[col].mode().iloc[0]))
                return self
                
    def plot_count_values(self, column: str, size: tuple = (10, 7)):
        """Plots the distribution of a column in the DataFrame."""
        plot_count_values(self, column, size)

    def plot_distribution(self, column: str, size: tuple = (10, 7)):
        """Plots the distribution of a column in the DataFrame."""
        plot_distribution(self, column, size)
        

    def plot_null_values(self, heatmap: bool = True, size: tuple = (10, 7)):
        """Plots a heatmap showing the distribution of null values in the DataFrame."""
        if heatmap:
            plot_null_values_heatmap(self, size)
        else:
            plot_null_values_bar(self, size)

    def train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self, test_size=test_size, random_state=random_state)
        return self.X_train, self.X_test, self.y_train, self.y_test



def read_csv(path: str):
    return DataFrame(pd.read_csv(path))

def read_excel(path: str):
    return DataFrame(pd.read_excel(path))

def read_json(path: str):
    return DataFrame(pd.read_json(path))

def read_parquet(path: str):
    return DataFrame(pd.read_parquet(path))

