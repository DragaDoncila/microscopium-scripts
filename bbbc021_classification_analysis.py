from matplotlib.colors import ListedColormap
from sklearn import neighbors
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

import itertools
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PATH = "/home/draga/FIT2082/Data_CSVs/"
OUT_PATH = "/home/draga/FIT2082/test/"


def main():
    all_accuracies = []
    all_labels = []

    # for each Data CSV available
    for filename in os.listdir(PATH):
        pattern = r"(Data2*)_*(.*)(_group).csv"

        match = re.search(pattern, filename)
        if match:
            print(match.groups())
            # load all data points from CSV
            point_data_full = pd.read_csv(PATH + filename)

            # split data into points with labelled and unlabelled MOA
            no_nones = point_data_full.loc[point_data_full['group'] != 'NONE']
            nones = pd.concat([point_data_full, no_nones]).drop_duplicates(keep=False)

            # generate dummy integer values for the group column
            group_dummies = pd.get_dummies(no_nones['group']).values.argmax(1)
            no_nones['group.dummy'] = group_dummies

            # get equal(ish) samples of each category
            sample_sizes = get_sample_sizes(no_nones, 30)
            no_nones = get_sampled_data(no_nones, group_dummies, sample_sizes)

            # define x and y values for plotting and classifying
            x_points = no_nones['x'].values
            y_points = no_nones['y'].values
            X = np.stack((x_points, y_points), axis=1)
            y = no_nones['group.dummy'].values

            # find best model for this data
            k, accuracies = get_best_k(X, y)

            # track accuracies and labels for all datasets
            label = (match.group(2) if len(match.group(2)) != 0 else "PCA") + (
                "_all_objects" if len(match.group(1)) != 5 else "_sampled")
            all_labels.extend([label for _ in range(len(accuracies))])
            all_accuracies.extend(accuracies)

            # fit classifier to data with value of k from CV
            clf = neighbors.KNeighborsClassifier(k, weights='distance')
            clf.fit(X, y)
            y_pred = cross_val_predict(clf, X, y, cv=5)
            scores = cross_val_score(clf, X, y, cv=5)

            # plot confusion matrix
            plot_matrix(y, y_pred, scores, k, match)

            plot_decision_boundaries(clf, X, no_nones, nones, match, k, np.mean(scores))

    # plot accuracies from k-nearest neighbours cross-validation
    plot_knearest_cv_results(all_accuracies, all_labels)


def get_counts(full_data):
    """
    Get counts of each group category in the dataset

    Parameters
    ----------
    full_data: pd DataFrame
        Data to search for categories

    Returns
    -------
    count_df: pd DataFrame
        Dataframe containing each unique category, its dummy value and the count of its occurrences in full_data
    """
    categories = list(np.unique(full_data['group']))
    dummies = []
    counts = []
    for info in categories:
        dummies.append(full_data.loc[full_data['group'] == info, 'group.dummy'].iloc[0])
        counts.append(full_data.loc[full_data['group'] == info, 'group'].agg(['count']).iloc[0])
    categories = pd.DataFrame(categories)
    counts = pd.DataFrame(counts)
    dummies = pd.DataFrame(dummies)
    count_df = pd.concat([categories, dummies, counts], axis=1)
    count_df.columns = ['Category', 'Category.Dummy', 'Count']
    return count_df


def get_sampled_data(data, y, sample_sizes):
    """
    Use random under sampling to sample data. Each category in y will be sampled according to its corresponding value
    in sample_sizes

    Parameters
    ----------
    data: pd DataFrame
        Data to be sampled
    y: np array
        Target categories to use for under sampling
    sample_sizes: dict
        {category_dummy: desired_count} for each category in data

    Returns
    -------
    x_resample: pd DataFrame
        resampled data
    """
    us = RandomUnderSampler(random_state=42, ratio=sample_sizes)

    # convert dataframe to numpy array
    col_names = data.columns
    col_types = data.dtypes
    data = data.values.astype("U")

    x_resample, y_resample = us.fit_sample(data, y)

    # convert back to dataframe
    x_resample = pd.DataFrame(x_resample)
    x_resample.columns = col_names
    for col_name, col_type in zip(col_names, col_types):
        x_resample[col_name] = x_resample[col_name].astype(col_type)

    return x_resample


def get_sample_sizes(data, desired_n):
    """
    Return a dictionary of sample sizes for each category in data to use for under sampling
    Will return the category count for categories with fewer than desired_n samples

    Parameters
    ----------
    data: pd DataFrame
        Data to get sample sizes for
    desired_n: int
        Desired number of points in each category

    Returns
    -------
    count_dict: dictionary
        {category dummy : category count} for each category in data

    """
    counts = get_counts(data)
    cat_list = list(counts['Category.Dummy'])
    count_list = []
    for cat in cat_list:
        count_list.append(min(counts.loc[counts['Category.Dummy'] == cat, 'Count'].iloc[0], desired_n))
    count_dict = dict(zip(cat_list, count_list))
    return count_dict


def get_best_k(X, y):
    """
    Perform cross validation on fitting X using KNN classifier with k = 3 to k = 21. Returns k which results in maximum
    mean accuracy after cross validation

    Parameters
    ----------
    X: np array
        Data to perform fit on
    y: np array
        Target values

    Returns
    -------
    (k, accuracy): tuple
        The value of k which gave the best mean accuracy score, and that accuracy

    """
    mean_accuracies = []
    all_accuracies = []
    for n_neighbors in range(3, 21):
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        scores = cross_val_score(clf, X, y, cv=5)
        mean_accuracy = np.mean(scores)
        # sd = np.std(scores)
        # print("Standard Deviation for K = {}, Best K = 10 : {}".format(n_neighbors, sd))
        all_accuracies.extend(scores)
        mean_accuracies.append(mean_accuracy)
    best_accuracy = max(mean_accuracies)
    best_k = mean_accuracies.index(best_accuracy) + 3

    return (best_k, all_accuracies)


def plot_decision_boundaries(classifier, X, no_nones, nones, match, k, accuracy):
    """
    Plot classified training points against decision map of K nearest neighbours classification and unknown MOA points

    Parameters
    ----------
    classifier: K-nearest model
        Classifier set up with best value of K from cross-validation
    X: 2D np-array
        Training data for classifier as (x,y) coordinates
    no_nones: pd dataframe
        Information about sampled training points
    nones: pd dataframe
        Information about unknown MOA points
    match: re match objects
        Match object containing filename of data
    k: integer
        Value of K used in classifier
    accuracy: float
        Average accuracy of classifier in cross-validation

    Returns
    -------
    None

    """
    # Define colour maps for mesh and scatter
    cmap_light = ListedColormap(sns.color_palette('pastel', 13))
    colours_bold = list(sns.color_palette('bright', 13))
    cats = np.unique(no_nones['group'])
    color_dict = dict(zip(cats, colours_bold))

    # Plot decision boundaries as a mesh
    h = 0.05  # step size in the mesh (.02 initially caused memory error)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, zorder=1)

    # Plot the unknown MOA with transparency
    ax.scatter(nones['x'].values, nones['y'].values, c=(135 / 255, 135 / 255, 135 / 255, 0.15), s=15, zorder=2,
               label="UNKNOWN")

    # Plot the classifier training set
    j = 0
    for i, dff in no_nones.groupby("group"):
        ax.scatter(dff['x'], dff['y'], s=20, c=color_dict[i],
                   edgecolors='k', label=i, zorder=3)
        j += 1

    # Ensure axis limits don't stretch beyond colour mesh
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    # Get name of algorithm being used for current dataset
    algorithm_name = "PCA" if len(match.group(2)) == 0 else match.group(2)
    sampling_status = "All Objects" if len(match.group(1)) == 4 else "Sampled Objects"

    # Set axis labels, title and legend
    ax.set_xlabel("{} Component 1\nk={} ;accuracy={:.4f} ".format(algorithm_name, k, accuracy))
    ax.set_ylabel("{} Component 2".format(algorithm_name))

    title = ("{} Embedding of BBBC021 Images \nwith KNN Classifier Decision Boundary, {}".format(algorithm_name,
                                                                                                 sampling_status))
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1.00, 0.4))

    # plt.savefig(OUT_PATH + "{}_final.png".format(algorithm_name))
    plt.show()


def plot_knearest_cv_results(all_accuracies, all_labels):
    """
    Plot distribution of accuracy of different values of K in cross validation, with 95% CI error bars,
    for different embedding algorithms

    Parameters
    ----------
    all_accuracies: python list of floats
        accuracies of each value of k tried during cross validation across different embedding algorithms
    all_labels: python list of strings
        embedding algorithm and sampling method belonging to each accuracy

    Returns
    -------
    None
    """

    # Create list to assign value of k to each plot point
    k_list = []
    for i in range(6):
        for j in range(3, 21):
            for k in range(5):
                k_list.append(j)
    k_list.extend(list(range(3, 21)))

    # Add baseline accuracy points
    all_accuracies.extend([1 / 13 for _ in range(3, 21)])
    all_labels.extend(['Baseline' for _ in range(3, 21)])

    # Create dataframe from k values, their accuracy, and the embedding method used
    accuracy_df = pd.DataFrame(k_list)
    accuracy_df.columns = ["k"]
    accuracy_df['Accuracy'] = all_accuracies
    accuracy_df['Method'] = all_labels

    # Plot the lines, with 95% confidence interval
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(x="k", y="Accuracy",
                 hue="Method",
                 data=accuracy_df, ci=95)

    # Make baseline accuracy dashed
    ax.lines[6].set_linestyle("--")

    # Move axis to make room for legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Define labels and symbols to appear in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    handles[-1].set_linestyle('dashed')
    labels = [labels[0], labels[1], labels[6], labels[4], labels[3], labels[5], labels[2], labels[7]]
    handles = [handles[0], handles[1], handles[6], handles[4], handles[3], handles[5], handles[2], handles[7]]

    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.title(
        "Distribution of Prediction Accuracy vs. Value of K\nin KNN Classification of Different Embedding Methods")
    plt.show()


def plot_matrix(y, y_pred, scores, k, match):
    """
    Calculate and plot confusion matrix for the given value of k using the actual and predicted y values.

    Parameters
    ----------
    y: ndarray
        Actual y values used in classifier
    y_pred: ndarray
        Predicted y values from classifier
    scores: ndarray
        Accuracy scores of given value of k
    k: integer
        Value of k used for this data
    match: re match objects
        Match corresponding to file name of data

    Returns
    -------
    None
    """
    cnf_matrix = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=2)

    plt.figure()
    plot_confusion_matrix(cnf_matrix,
                          classes=["Actin distruptors", "Aurora kinase inhibitors", "Cholesterol-lowering", "DMSO",
                                   "DNA damage", "DNA replication", "Eg5 inhibitors", "Epithelial", "Kinase inhibitors",
                                   "Microtubule destabilizers", "Microtubule stabilizers", "Protein degradation",
                                   "Protein synthesis"],
                          title='Mean Prediction of Treatment Mode of Action in 5-fold CV on KNN Classifier\n {} Embedding, {}'.format(
                              "PCA" if len(match.group(2)) == 0 else match.group(2),
                              "All Objects" if len(match.group(1)) == 4 else "Sampled Objects"))
    plt.xticks(rotation=82)
    plt.xlabel(
        'Predicted label\naccuracy={:0.4f}; misclass={:0.4f}\n k={}'.format(np.mean(scores), 1 - np.mean(scores), k))
    plt.show()


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.

    Parameters
    ----------
    cm: ndarray
        Array of counts classified in each predicted vs. actual class
    classes: python list of strings
        Labels of each class
    title: string
        Desired title of plot. Default is 'Confusion Matrix'
    cmap: LinearSegmentedColormap
        Colour map to use for confusion matrix. Default is Blues.

    Returns
    -------
    None
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


main()
