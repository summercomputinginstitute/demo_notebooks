# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# <a href='https://ai.meng.duke.edu'> = <img align="left" style="padding-top:10px;" src=https://storage.googleapis.com/aipi_datasets/Duke-AIPI-Logo.png>

# # Decision Trees & Random Forests

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
# -

# ## Tree models

# Let's start with a simple example to understand how the depth of the tree impacts overfitting/underfitting.  We will generate some synthetic data which has two features and belongs to 4 classes.  We will then illustrate how to classify the datapoints using a tree model.

# +
# Generate some synthetic data
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)

plt.figure(figsize=(10,6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')
plt.show()


# -

# Function to visualize the decision boundaries created by a classification model
def plot_decision_boundaries(X,y,model):
    
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    markers = ['^','s','v','o','x']
    colors = ['yellow','green','purple','blue','orange']
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    for i,k in enumerate(np.unique(y)):
        plt.scatter(X.loc[y==k].iloc[:,0],X.loc[y==k].iloc[:,1],
                    c=colors[i],marker=markers[i],label=k,edgecolor='black')

    xgrid = np.arange(X.iloc[:,0].min(),X.iloc[:,0].max(),
                      (X.iloc[:,0].max()-X.iloc[:,0].min())/500)
    ygrid = np.arange(X.iloc[:,1].min(),X.iloc[:,1].max(),
                      (X.iloc[:,1].max()-X.iloc[:,1].min())/500)
    xx,yy = np.meshgrid(xgrid,ygrid)
    
    mesh_preds = model.predict(np.c_[xx.ravel(),yy.ravel()])
    mesh_preds = mesh_preds.reshape(xx.shape)
    plt.contourf(xx,yy,mesh_preds,alpha=0.2,cmap=cmap)
    plt.legend()
    return


# We will now build a decision tree model to classify the data.  We will set `max_depth=None`, meaning that our tree can grow as deep as it would like in order to correctly classify the training data.

# +
# Instantiate the decision tree model using Scikit-Learn's DecisionTreeClassifier()
tree_model = DecisionTreeClassifier(criterion='gini',max_depth=None,min_samples_leaf=1,random_state=0)

# Fit the tree to the data
tree_model.fit(X, y)

# Plot the decision boundaries of the tree model
plt.figure(figsize=(10,6))
plot_decision_boundaries(X,y,tree_model)
# -

# As we can see above, our model appears to be overfitting the data - for example, take a look at the small yellow rectangle between the red and blue areas, or the small red area in the midst of the yellow.  It's doubtful that our actual decision boundary between classes looks like this - what is likely happening here is that our tree model is overfitting to single points. Let's run again using a fixed value of 2 for max_depth and look at the resulting decision boundaries

# +
# Instantiate the decision tree model using max_depth=2
tree_model = DecisionTreeClassifier(criterion='gini',max_depth=2,min_samples_leaf=1,random_state=0)

# Fit the tree to the data
tree_model.fit(X, y)

# Plot the decision boundaries of the tree model
plt.figure(figsize=(10,6))
plot_decision_boundaries(X,y,tree_model)
# -

# Here we can see that this model is also not ideal.  Because we have constrained the tree depth to 2 layers, our model is too simple and is only able to classify the data into 3 classes rather than the 4 we actually have.  Let's try one more time, setting our depth to 3 layers.

# +
# Instantiate the decision tree model using max_depth=3
tree_model = DecisionTreeClassifier(criterion='gini',max_depth=3,min_samples_leaf=1,random_state=0)

# Fit the tree to the data
tree_model.fit(X, y)

# Plot the decision boundaries of the tree model
plt.figure(figsize=(10,6))
plot_decision_boundaries(X,y,tree_model)
# -

# This looks better! We can see that as we change max_depth, we get a simpler or more complex model.  We can tune this as we do with any hyperparameter and find the optimal depth of our tree using a validation set or cross-validation to ensure we are not overfitting our tree to the training data, which is a common risk when working with single decision trees.

# ### Visualizing a decision tree
# Let's do another example using a decision tree model, and this time we are going to use Scikit-Learn's functionality to visualize the decision tree that gets created.  For this example we will again be working with the 'iris' dataset.

# +
# Load the dataset using a helper function in Seaborn
iris = sns.load_dataset('iris')

# Create feature matrix
X = iris.drop(labels='species',axis=1)

# Create target vector
y = iris['species']

# Let's set aside a test set and use the remainder for training and cross-validation
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0,test_size=0.25)

X.head()

# +
# Instantiate the decision tree model
tree_model = DecisionTreeClassifier(criterion='gini',max_depth=None,min_samples_leaf=1,random_state=0)

# Fit the tree to the data
tree_model.fit(X_train, y_train)
# -

# Visualize the decision tree
plt.figure(figsize=(15,15))
plot_tree(tree_model,feature_names=X.columns,class_names=y.unique(),filled=True)
plt.show()

# The above diagram of our tree shows us a wealth of information about the model.  For each node in the tree we can see the splitting criteria it uses at that node (feature and value) to create the children nodes.  We can also see a measure of the impurity of the data at that point in the tree (gini impurity) as well as the number of datapoints (samples) at the node from our training data.  Finally, we can see the count of points belonging to each of the 3 classes (value) and a predicted class (class) for the node - the shading of the node also represents the predicted class of the node based on the most common class among the points at the node.

# Calculate the accuracy on the test set
test_preds = tree_model.predict(X_test)
test_acc = np.sum(test_preds==y_test)/len(y_test)
print('Test set accuracy is {:.3f}'.format(test_acc))

# Let's run one more and we will limit the min_samples_leaf to 5 samples per leaf of the tree to build a smaller tree and reduce the possibility of overfitting

# +
# Instantiate the decision tree model
tree_model = DecisionTreeClassifier(criterion='gini',max_depth=None,min_samples_leaf=5,random_state=0)

# Fit the tree to the data
tree_model.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(8,8))
plot_tree(tree_model,feature_names=X.columns,class_names=y.unique(),filled=True)
plt.show()
# -

# Now we can see that our model has produced a smaller tree, by limiting the depth to which the tree can grow.

# Calculate the accuracy on the test set
test_preds = tree_model.predict(X_test)
test_acc = np.sum(test_preds==y_test)/len(y_test)
print('Test set accuracy is {:.3f}'.format(test_acc))

# ## Random Forests

# For tackling real-world problems we will generally use a Random Forest ensemble model rather than a single decision tree to reduce the probability of overfitting. Let's do an example of a Random Forest using the breast cancer dataset we have worked with earlier from the University of Wisconsin.

# +
# Load the data
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer(as_frame=True)
X,y=data.data,data.target
# Since the default in the file is 0=malignant 1=benign we want to reverse these
y=(y==0).astype(int)

# Let's set aside a test set and use the remainder for training and cross-validation
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0,test_size=0.4)

X_train.head()
# -

# Let's start with a simple decision tree model to get a baseline.

# +
# Instantiate the decision tree model
tree_model = DecisionTreeClassifier(criterion='gini',max_depth=None,min_samples_leaf=1,random_state=0)

# Fit the tree to the data
tree_model.fit(X_train, y_train)

test_preds = tree_model.predict(X_test)
test_acc = np.sum(test_preds==y_test)/len(y_test)
print('Test set accuracy is {:.3f}'.format(test_acc))
# -

# Now we'll try a random forest model, and use the default settings from scikit-learn for the hyperparameters of the model.

# +
# Instantiate the random forest model
rf_model = RandomForestClassifier(criterion='gini',max_depth=None, min_samples_leaf=1,n_estimators=100,
                                 max_features='auto',max_samples=None,random_state=0)

# Fit the random forest to the data
rf_model.fit(X_train, y_train)

test_preds = rf_model.predict(X_test)
test_acc = np.sum(test_preds==y_test)/len(y_test)
print('Test set accuracy is {:.3f}'.format(test_acc))
# -

# Let's build another Random Forest model which will include all features in every tree that we grow, but reduce the number of training rows to 0.7 (70% of the total rows of the training set will be selected for use in any given tree in the forest).

# +
# Instantiate the random forest model
rf_model = RandomForestClassifier(criterion='gini',max_depth=None, min_samples_leaf=1,n_estimators=1000,
                                 max_features=1.,max_samples=0.7,random_state=0)

# Fit the random forest to the data
rf_model.fit(X_train, y_train)

test_preds = rf_model.predict(X_test)
test_acc = np.sum(test_preds==y_test)/len(y_test)
print('Test set accuracy is {:.3f}'.format(test_acc))
# -

# Finally, let's build one more model which simplifies the trees by setting min_samples_leaf=5 and reducing max_features to 0.1 (any given tree uses only 10% of the features in the data), but increase the number of trees to 1000

# +
# Instantiate the random forest model
rf_model = RandomForestClassifier(criterion='gini',max_depth=None, min_samples_leaf=5,n_estimators=1000,
                                 max_features=0.1,max_samples=0.7,random_state=0)

# Fit the random forest to the data
rf_model.fit(X_train, y_train)

test_preds = rf_model.predict(X_test)
test_acc = np.sum(test_preds==y_test)/len(y_test)
print('Test set accuracy is {:.3f}'.format(test_acc))
# -

# In this case, a larger number of simpler trees beats a smaller number of more complex trees (this is not always the case, but often!)
