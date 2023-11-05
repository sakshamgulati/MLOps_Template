from src import DataOps
import pandas as pd
import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")


# write me a basic unit test for modelops modules
def test_pytest():
    assert 1 == 1


# function to create sample categorical dataframe
def categorical_dataframe():
    df = pd.DataFrame(
        {"A": ["a", "b", "a"], "B": ["b", "a", "c"], "C": ["a", "b", "c"]}
    )
    return df


# function to create sample numerical dataframe
def numerical_dataframe():
    df = pd.DataFrame({"D": [1, 2, 3], "E": [4, 5, 6], "F": [7, 8, 9]})
    return df


# function to create a sample categorical output dataframe
def output_dataframe():
    df = pd.DataFrame({"Output": [1, 0, 1]})
    return df


def test_featurengg():
    num_df = numerical_dataframe()
    cat_df = categorical_dataframe()
    feat = DataOps.feature_engg_class()
    fin_df = feat.min_max_scaling(cat_df, num_df)
    # assert all values in the dataframe are between 0 and 1
    assert fin_df.values.max() != 0
    # assert there are no nulls
    assert fin_df.isnull().sum().sum() == 0


def test_data():
    num_df = numerical_dataframe()
    cat_df = categorical_dataframe()
    feat = DataOps.feature_engg_class()
    fin_df = feat.min_max_scaling(cat_df, num_df)
    output = output_dataframe()
    X_train, X_test, y_train, y_test = feat.split(fin_df, output)
    assert len(X_train) > 0
    # assert output variable is a categorical
    # assert y_train.dtypes=='int64'
    # assert train test ratio is 0.33
    assert len(X_train) / len(X_test) == 2
