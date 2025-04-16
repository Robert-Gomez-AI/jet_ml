import pandas as pd
from .plots.plot_configs import *
from sklearn.model_selection import train_test_split
from .plots.plots import *
from typing import List, Dict, Tuple



class DataFrame(pd.DataFrame):
    """
    A class that extends pandas.DataFrame with additional methods for data preprocessing.

    Attributes:
        n (int): Number of rows in the DataFrame.
        p (int): Number of columns in the DataFrame.
        numerical_dataframe (pd.DataFrame): DataFrame containing only numerical columns.
        categories_dict (Dict[str, Dict[str, int]]): Dictionary containing the categories of the columns.
    Methods:
        set_categories_dict(categories_dict: Dict[str, Dict[str, int]]): Sets the categories dictionary.
        transform_to_numerical(columns_to_convert: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]: Transforms the DataFrame to a numerical DataFrame.
        get_numerical_dataframe() -> pd.DataFrame: Returns the numerical DataFrame.
        get_categories_dict() -> Dict[str, Dict[str, int]]: Returns the categories dictionary.
        null_values_treatment(method: str, column: list[str] = None): Treats null values in the DataFrame based on the specified method.
        plot_count_values(column: str, show_probability: bool = False, size: tuple = (10, 7)): Plots the distribution of a column in the DataFrame.
        plot_distribution(column: str, size: tuple = (10, 7)): Plots the distribution of a column in the DataFrame.
        plot_null_values(heatmap: bool = True, size: tuple = (10, 7)): Plots a heatmap showing the distribution of null values in the DataFrame.
        train_test_split(numerical:bool = False ,test_size: float = 0.2, random_state: int = 42): Splits the DataFrame into training and testing sets.
        
    """

    @property
    def _constructor(self):
        """
        Constructor property needed for pandas inheritance
        """
        return DataFrame
    
    def __new__(cls, *args, **kwargs):
        """
        Create a new DataFrame object
        """
        return super(DataFrame, cls).__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = self.shape[0]
        self.p = self.shape[1]
    
    def __finalize__(self, other, method=None, **kwargs):
        """
        Propagate metadata from other to self
        """
        for name in ['n', 'p', 'numerical_dataframe', 'categories_dict']:
            if hasattr(other, name):
                object.__setattr__(self, name, getattr(other, name))
        return self


    def set_categories_dict(self, categories_dict: Dict[str, Dict[str, int]]):
        self.categories_dict=categories_dict
    def transform_to_numerical(self, columns_to_convert: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        self.numerical_dataframe, self.categories_dict=one_hot_encoding_dataframe(self,columns_to_convert=columns_to_convert)
    def get_numerical_dataframe(self):
        return self.numerical_dataframe
    
    def get_categories_dict(self):
        return self.categories_dict

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
        if column is not None:
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
                
    

    def train_test_split(self, numerical:bool = False ,test_size: float = 0.2, random_state: int = 42):
        if numerical:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.numerical_dataframe, test_size=test_size, random_state=random_state)
        
        return self.X_train, self.X_test, self.y_train, self.y_test



class GraficalDataFrame(DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_count_values(self,column: str,show_probability: bool = False,  size: tuple = (10, 7)):
        """Plots the distribution of a column in the DataFrame."""
        plot_count_values(self,column, size, show_probability=show_probability)

    def plot_distribution(self, column: str, size: tuple = (10, 7)):
        """Plots the distribution of a column in the DataFrame."""
        plot_distribution(self, column, size)
        

    def plot_null_values(self, heatmap: bool = True, size: tuple = (10, 7)):
        """Plots a heatmap showing the distribution of null values in the DataFrame."""
        if heatmap:
            plot_null_values_heatmap(self, size)
        else:
            plot_null_values_bar(self, size)



def one_hot_encoding(column):
    categories = list(set(column))
    print(f'Hay {len(categories)} categorias')
    
    if len(categories) > 3:
        # Para columnas con más de 3 categorías, usar one-hot encoding
        result = []
        category_dict = {}
        for i, category in enumerate(categories):
            # Crear vector one-hot para cada categoría
            one_hot = [1 if x == category else 0 for x in column]
            result.append(one_hot)
            category_dict[category] = i
        return result, category_dict
    else:
        # Para columnas con 2-3 categorías, usar encoding ordinal
        category_dict = {cat: i for i, cat in enumerate(categories)}
        result = [category_dict[x] for x in column]
        return result, category_dict

def one_hot_encoding_dataframe(df: DataFrame, columns_to_convert: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    numeric_df = df.copy()
    categories_dict = {}
    
    for column in columns_to_convert:
        values, category_mapping = one_hot_encoding(list(numeric_df[column]))
        
        if isinstance(values[0], list):  # Si es one-hot encoding
            for i, category in enumerate(category_mapping.keys()):
                numeric_df[f"{column}_{category}"] = values[i]
            numeric_df.drop(column, axis=1, inplace=True)
        else:  # Si es encoding ordinal
            numeric_df[column] = values
            
        categories_dict[column] = category_mapping
    numeric_df= DataFrame(numeric_df)
    numeric_df.set_categories_dict(categories_dict)
    return numeric_df, categories_dict


def read_csv(path: str):
    return DataFrame(pd.read_csv(path))

def read_excel(path: str):
    return DataFrame(pd.read_excel(path))

def read_json(path: str):
    return DataFrame(pd.read_json(path))

def read_parquet(path: str):
    return DataFrame(pd.read_parquet(path))

