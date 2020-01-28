import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.utils.utils import ecdf


def plot_pca_components(df):
    df_pca_comp = pd.DataFrame(df.var(), columns=["variance"])
    df_pca_comp_sum = df.var().sum()
    df_pca_comp["ratio"] = df_pca_comp.apply(lambda x: x / df_pca_comp_sum)
    df_pca_comp["index"] = list(range(len(df_pca_comp)))

    for i in range(5, 29, 5):
        print(f"{i} components ratio : ", df_pca_comp["ratio"][0:i].sum())

    ax = df_pca_comp.plot(x="index", y="variance", style='.-', figsize=(15, 5), grid=True)
    ax.set_title("PCA components ratio")
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Variance (σ²)")

    plt.savefig("../reports/figures/1.PCA_components.png")
    plt.show()


def plot_frauds_counter(df,):
    plt.figure(figsize=(15, 5))
    ax = sns.countplot(x="Class", data=df)
    ax.set_title("Fraud counter")
    ax.grid()

    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2.,
                height + 100,
                '{:1.1f}'.format(height),
                ha="center", fontsize=15)

    plt.savefig("../reports/figures/2.Frauds_counter.png")
    plt.show()


def plot_feature_distribution(df, features, label1='0', label2="1"):
    df1 = df.loc[df['Class'] == 0]
    df2 = df.loc[df['Class'] == 1]
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(6, 5, figsize=(18, 22))

    for i, feature in enumerate(features):
        plt.subplot(6, 5, i + 1)
        sns.distplot(df1[feature], hist=False, label=label1)
        sns.distplot(df2[feature], hist=False, label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)

    plt.suptitle("Feature distribution", fontsize=12)

    plt.savefig("../reports/figures/3.feature distribution.png")
    plt.show()

def plot_traingled_corr_mat(df):
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 20))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True,fmt=".1%")

    plt.savefig("../reports/figures/4.Corelation_matrix.png")
    plt.show()

def plot_transactions_scatter(df):
    fig, ax = plt.subplots(1, 2, figsize=(15,5), sharex=True)
    fig.suptitle('Time of transaction vs Amount', fontsize=16)
    colors = ["#0101DF", "#DF0101"]

    ax[0].scatter(df[df['Class']==1].Time, df[df['Class']==1].Amount)
    ax[0].set_title('Fraud')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amount')

    ax[1].scatter(df[df['Class']==0].Time, df[df['Class']==0].Amount)
    ax[1].set_title('Normal')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Amount')

    plt.savefig("../reports/figures/5.Transactions_scatter.png")
    plt.show()

def plot_transactions_historagram(df):
    fraud = df[df["Class"] == 1]
    normal = df[df["Class"] == 0]

    f, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))
    f.suptitle('Number of transactions by amount',size = 16)

    ax1.hist(fraud.Amount, bins = 100)
    ax1.set_title(f'Fraud (min : {fraud.Amount.min()} | max : {fraud.Amount.max()} )')
    ax1.set_xlabel('Amount ($)')
    ax1.set_ylabel('Number of Transactions')
    ax1.set_yscale("log")

    ax2.hist(normal.Amount, bins = 100)
    ax2.set_title(f'Normal (min : {normal.Amount.min()} | max : {normal.Amount.max()} )')
    ax2.set_xlabel('Amount ($)')
    ax2.set_ylabel('Number of Transactions')
    ax2.set_yscale("log")

    plt.savefig("../reports/figures/6.Transactions_histogram.png")
    plt.show()


def plot_transactions_density(df):
    normal = df.loc[df['Class'] == 0]["Time"]
    fraud = df.loc[df['Class'] == 1]["Time"]
    plt.figure(figsize = (14,4))
    plt.title('Credit Card Transactions Time Density Plot')
    sns.distplot(fraud,kde=True,bins=10,label = "Fraud")
    sns.distplot(normal,kde=True,bins=10,label="Normal")

    plt.xlabel("time(sample)")
    plt.ylabel("Density")
    plt.legend()

    plt.savefig("../reports/figures/7.Transactions_density.png")
    plt.show()

# Plot EDF normal


def plot_ecdf(df):
    normal = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]

    fig, ax = plt.subplots(1,2,figsize=(15,5))

    x_amount_fraud , y_amount_fraud = ecdf(fraud["Amount"].values)
    ax[0].plot(x_amount_fraud ,y_amount_fraud,marker= ".",linestyle = 'none')
    ax[0].set_xlabel("Amount $")
    ax[0].set_ylabel("Cumulative probability")
    ax[0].set_title("Fraud")

    x_amount_norm , y_amount_norm = ecdf(normal["Amount"].values)
    ax[1].plot(x_amount_norm ,y_amount_norm,marker= ".",linestyle = 'none')
    ax[1].set_xlim(0,2200)
    ax[1].set_xlabel("Amount $")
    ax[1].set_ylabel("Cumulative probability")
    ax[1].set_title("Normal")

    plt.suptitle("Cumulative probability")
    plt.show()