import pandas as pd

def load_data(path: str) ->  pd.DataFrame:
    #Loan the data from a CSV file
    return pd.read_csv(path)