import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

def plot_relational_plot(df):
    """Creates and saves a relational scatter plot."""
    fig, ax = plt.subplots()
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], ax=ax)
    ax.set_title("Relational Plot")
    plt.savefig('relational_plot.png')
    return

def plot_categorical_plot(df):
    """Creates and saves a categorical bar plot."""
    fig, ax = plt.subplots()
    df.iloc[:, 2].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Categorical Plot")
    plt.savefig('categorical_plot.png')
    return

def plot_statistical_plot(df):
    """Creates and saves a statistical correlation heatmap."""
    fig, ax = plt.subplots()

    # Only use numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])

    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Statistical Plot - Correlation Heatmap")
    plt.savefig('statistical_plot.png')
    return


def statistical_analysis(df, col: str):
    """Computes the four main statistical moments of a given column."""
    mean = np.mean(df[col])
    stddev = np.std(df[col])
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis

def preprocessing(df):
    """Preprocesses the dataset: handles missing values and provides insights."""
    df = df.dropna()  # Remove missing values

    print(df.describe())  # Summary statistics

    # Only select numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=[np.number])  
    print(numeric_df.corr())  # Now correlation works without error

    return df


def writing(moments, col):
    """Prints the statistical moments with interpretations."""
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    if -2 < moments[2] < 2:
        skew_desc = "not skewed"
    else:
        skew_desc = "right skewed" if moments[2] > 2 else "left skewed"
    
    if moments[3] > 0:
        kurtosis_desc = "leptokurtic"
    elif moments[3] < 0:
        kurtosis_desc = "platykurtic"
    else:
        kurtosis_desc = "mesokurtic"
    
    print(f'The data was {skew_desc} and {kurtosis_desc}.')
    return

def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = df.columns[1]  # Select second column for analysis
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return

if __name__ == '__main__':
    main()
