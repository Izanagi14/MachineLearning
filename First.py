import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
fruits = pd.read_table('fruit_data_with_colors.txt')
print fruits.head()
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique()))
print lookup_fruit_name
print lookup_fruit_name[2]



from matplotlib import cm
X = fruits[['height', 'width', 'mass']]
#X = fruits[['height', 'width', 'mass', 'color_score']]
Y = fruits['fruit_label']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=0)
cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X_train,c=Y_train,marker='o',s=40,hist_kwds={'bins':15},figsize=(9,9),cmap=cmap)


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X_train['width'],X_train['height'],X_train['mass'],c = Y_train, marker = 'o',s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
#plt.show()



#Identify the Fruits according the height mass width
# the Y label is the fruits label the dictionary contains the label with name we need the label correctly identified then we can state the name


#now using the Knearest Neighbour


	
from sklearn.neighbors import KNeighborsClassifier
knn =  KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,Y_train)
print knn.score(X_test,Y_test)

testing  = np.array([20,4.3,5.5])

testing = testing.reshape(1,-1)

print X_test
fruit_prediction = knn.predict(testing)
	


print lookup_fruit_name[fruit_prediction[0]]
