import sklearn as sk
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
import seaborn as sns

def prepare_data():
    X, y = load_wine(return_X_y=True) # X contains values for attributes, y contains class labels
    # rearrange dataset
    X = X[:,:2] # now X is a 2D dataset
    # create samples
    validate_size, test_size = round(len(X) * 0.2), round(len(X) * 0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    scaler = StandardScaler()
    scaler.fit(X_train) # here validation is in the whole dataset so we can fit scalar properly
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=validate_size, random_state=1)
    X_train = scaler.transform(X_train)
    X_validate=scaler.transform(X_validate)
    X_test=scaler.transform(X_test)
    return X_train, y_train, X_validate, y_validate, X_test, y_test

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_sub(nr, nc, plots):
    # plot settings
    plt.style.use('seaborn-ticks')
    plt.rcParams['font.size'] = 20
    rows = nr
    columns = nc
    fig, axes = plt.subplots(rows,columns,figsize=[25,25])
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top= 0.8, wspace=0.1 , hspace=0.3)
    # assign a name to each ax
    d = {}
    i = 0
    for r in range(rows):
        for c in range(columns):
            d[i] = axes[r][c]
            i += 1
    for attr, value in plots.items():
        value.ax = d[attr]
    plt.show()


def get_knn_clf(X_train, y_train, k):
    neigh = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf = neigh.fit(X_train, y_train)
    return clf

def get_score(clf, X_validate, y_validate):
    return clf.score(X_validate, y_validate)

def plot_knn(X,y,neigh):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    Z = neigh.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)" % (neigh.n_neighbors))
    plt.show()

def plot_scores(objs, scores, title, color):
    y_pos = np.arange(len(objs))
    plt.xticks(y_pos, objs)
    plt.ylabel('Accuracy')
    plt.title(title)
    i = 0
    for score in scores:
        plt.text(i, score, "{0:.2f}".format(score))
        i = i + 1
    return plt.bar(y_pos, scores, align='center', alpha=0.5, color=color)

def do_knn(kn_values, X_train, y_train, X_validate, y_validate, plot=False):
    sc_count = 0
    scores = [0,0,0,0] #todo: size based on kn_values size
    max_score = 0
    best_clf = None
    for kn in kn_values:
        clf = get_knn_clf(X_train, y_train, kn)
        score = get_score(clf, X_validate, y_validate)
        scores[sc_count] = score
        sc_count += 1
        if score >= max_score:
            max_score = score
            best_clf = clf
        if plot is True:
            plot_knn(X_train, y_train, clf)
        else:
            print("score: %2f, k: %2d" %(score, kn))
    return best_clf, scores

def do_svm(kernel_type, C, X_train, y_train, X_validate, y_validate, g="auto", plot=False):
    svm_scores=[0,0,0,0,0,0,0]
    svm_max_score = 0
    best_svm_clf = None
    svm_sc_count = 0
    for c in C:
        if isinstance(g, str):
            model = svm.SVC(C=c, kernel=kernel_type, gamma=g)
        else:
            model = svm.SVC(C=c, kernel=kernel_type, gamma=g[0])
        svm_clf = model.fit(X_train, y_train)
        score = get_score(svm_clf, X_validate, y_validate)
        svm_scores[svm_sc_count] = score
        svm_sc_count += 1
        if isinstance(g, float) and plot is False:
            print("score: %2f, C: %2f, g: %2f" %(score, c, g))
        if isinstance(g, str) and plot is False:
            print("score: %2f, C: %2f, g: %s" %(score, c, g))
        if score > svm_max_score:
            svm_max_score = score
            best_svm_clf = svm_clf
        if plot is True:
            fig, ax = plt.subplots()
            # title for the plots
            title = ('SVC data and boundaries for C= %2f ' %(c))
            # Set-up grid for plotting.
            X0, X1 = X_train[:, 0], X_train[:, 1]
            xx, yy = make_meshgrid(X0, X1)
            plot_contours(ax, svm_clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(title)
            plt.show()
    return best_svm_clf, svm_scores

def plot_svm_grid(cs,gs,scores):
    df = pd.DataFrame({"param_C": cs, "param_gamma":gs, "scores":scores})
    pvt = pd.pivot_table(df, values='scores', index='param_C', columns='param_gamma')
    ax = sns.heatmap(pvt,annot = True)
    plt.title("Heatmap for hardcoded grid search")
    plt.show()

def do_svm_grid(kernel_type, C, X_train, y_train, X_validate, y_validate, gs, plot=False):
    svm_max_score = 0
    best_svm_clf = None
    svm_sc_count = 0
    cs = {}
    gammas = {}
    pvt_scores = {}
    count = 0
    for c in C:
        for g in gs:
            cs[count] = c
            gammas[count] = g
            model = svm.SVC(C=c, kernel=kernel_type, gamma=g)
            svm_clf = model.fit(X_train, y_train)
            score = get_score(svm_clf, X_validate, y_validate)
            pvt_scores[count] = score
            count = count + 1
            svm_sc_count += 1
            if plot is False:
                print("score: %2f, C: %2f, g: %2f" %(score, c, g))
            if score > svm_max_score:
                svm_max_score = score
                best_svm_clf = svm_clf
    if plot is True:
        plot_svm_grid(cs,gammas,pvt_scores)
    return best_svm_clf 

def plotKfold(clf, C, gamma):
    pvt = pd.pivot_table(pd.DataFrame(clf.cv_results_), values='mean_test_score', index='param_C', columns='param_gamma')
    ax = sns.heatmap(pvt,annot = True)
    plt.title("Heatmap for Grid search with 5-fold validation")
    plt.show()
 

def get_svm_fold(X_train, y_train, plot=False):
    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [1e-5, 1e-4, 1e-3, 1e-1, 10, 1000], 'C': [0.001, 0.01, 0.1, 1, 10, 100,1000]},
        {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100,1000]}
        ] 
    model = GridSearchCV(SVC(), tuned_parameters, cv=5, iid=False)
    clf = model.fit(X_train, y_train)
    if plot is False:
        print("Best parameters set found on development set: ", clf.best_params_)
    if plot is True:
        plotKfold(clf, [0.001, 0.01, 0.1, 1, 10, 100,1000], [0.001, 0.01, 0.1, 1, 10, 100,1000])
    return clf


