import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import random
class Kmeans:
    def __init__(self):
        self.data = []
        self.k = 3

    def read_data(self):
        self.data = pd.read_csv("../data.csv",header=0);
        self.preprocess_data()


    def preprocess_data(self):
        self.data.drop("id",axis=1,inplace=True)
        self.data.drop("Unnamed: 32",axis=1,inplace=True)
        self.analyse_data()

    def analyse_data(self):
        # print self.data.head()
        # print self.data.diagnosis.unique()
        # self.data['diagnosis'] = self.data['diagnosis'].map({'M': 1, 'B': 0})
        # plt.hist(self.data['diagnosis'])
        # plt.title("Diagnosis M:1 B:0")
        # plt.show()
        # print self.data.describe()

        # Split the data as X and labels as Y
        x=self.data.ix[:,1:].values
        y=self.data.ix[:,0].values

        #Uncomment this function for visualising each feature for both class
        # self.visualize_data(x,y)

        #standardise the value by dividing by mean
        x_std = StandardScaler().fit_transform(x)
        # print x_std

        #Here we are using covariance matrix for calculating eigenpairs
        mean_vec = np.mean(x_std, axis=0)
        cov_mat = (x_std - mean_vec).T.dot((x_std - mean_vec)) / (x_std.shape[0] - 1)

        #eigen values and eigen vector are found using numpy function
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)


        #create eigen values and eigen vector pairs as typles
        eig_pairs = [(i,np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=lambda x: x[1], reverse=True)

        #Measure explained variance (Variance of each features)
        tot = sum(eig_vals)
        var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)

        #Uncomment the following code to visualize the pca

        # with plt.style.context('seaborn-whitegrid'):
        #     plt.figure(figsize=(6, len(eig_vals)))
        #
        #     plt.bar(range(len(eig_vals)), var_exp, alpha=0.5, align='center',
        #             label='individual explained variance')
        #     plt.step(range(len(eig_vals)), cum_var_exp, where='mid',
        #              label='cumulative explained variance')
        #     plt.ylabel('Explained variance ratio')
        #     plt.xlabel('Principal components')
        #     plt.legend(loc='best')
        #     plt.tight_layout()
        #     plt.show()

        #from the above plot we can 13 features out of 30
        self.rd=[]
        for i in range(15):
            self.rd.append(x.T[eig_pairs[i][0]])
        # print self.data.head()
        self.rd = np.array(self.rd).T
        self.rd = StandardScaler().fit_transform(self.rd)
        self.initiate_clustering()

    def initiate_clustering(self):

        self.centroids=self.initialize_centroids();

        old_centroids = [[] for i in range(self.k)]

        iteration =0

        while not self.converged(old_centroids,self.centroids,iteration):
            iteration += 1
            clusters = [[] for i in range(self.k)]

            clusters = self.assign_points(self.centroids, clusters)

            index = 0
            for cluster in clusters:
                old_centroids[index] = self.centroids[index]
                self.centroids[index] = np.mean(cluster, axis=0).tolist()
                index += 1
        for i in range(len(clusters)):
            print "The number of data point in cluster ",i," is ",len(clusters[i])

    def converged(self,olcent,cent,iteration):
        max_iter=100;
        if iteration > max_iter:
            return True
        return olcent==cent

    def initialize_centroids(self):
        centroids=[]
        for i in range(self.k):
            centroids.append(self.rd[np.random.randint(0, len(self.rd), size=1)].flatten().tolist())
        return centroids

    def assign_points(self,centroids,clusters):
        for instance in self.rd:
            # based on the minimum value of euclidean distance between the point and centroid
            mu_index = min([(i[0], np.linalg.norm(instance - centroids[i[0]])) \
                            for i in enumerate(centroids)], key=lambda t: t[1])[0]

            try:
                clusters[mu_index].append(instance)
            except KeyError:
                clusters[mu_index] = [instance]

        for cluster in clusters:
            if not cluster:
                cluster.append(data[np.random.randint(0, len(self.rd), size=1)].tolist())
        return clusters



    def visualize_data(self,x,y):

        # get the header of features leaving the description
        header_list = list(self.data)[1:]
        print header_list

        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(8, len(header_list)))
            for i in range(len(header_list)):
                plt.subplot((len(header_list) / 2) + 1, 2, i + 1)
                for lab in ('M', 'B'):
                    plt.hist(x[y == lab, i],
                             label=lab,
                             bins=10,
                             alpha=0.3, )
                plt.xlabel(header_list[i])
            plt.legend(loc='upper right', fancybox=True, fontsize=8)

            plt.tight_layout()
            plt.show()

def main():
    model = Kmeans()
    model.read_data()

if __name__ == '__main__':
    main()