import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def basic_statistics(df):
    return df.describe()

def plot_pairplot(df, cols):
    sns.pairplot(df[cols])
    plt.show()
