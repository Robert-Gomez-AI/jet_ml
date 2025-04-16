from ..preprocess import DataFrame , GraficalDataFrame
import os
import subprocess
import zipfile

def load_breast_cancer(graphical=False):
    """
    Load breast cancer dataset from sklearn and return as DataFrame
    
    Parameters
    ----------
    graphical : bool, default=False
        If True, returns a graphical DataFrame, otherwise returns regular DataFrame
        
    Returns
    -------
    DataFrame
        The breast cancer dataset as a DataFrame
    """
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    if graphical:
        df = GraficalDataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    
    else:
        df = DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    

        
    return df
