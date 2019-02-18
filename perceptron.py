import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


class perceptron(object):

    def training(self,data,dim):

        self.data = data
        self.n = 1
        self.w = np.random.rand(dim)
        self.b = 0
        self.k = None
        self.R = np.amax([np.linalg.norm(element[:-1]) for element in self.data]) #R = max{norm({all_points}})
        step = 0

        while self.k != 0:

            self.k = 0
            
            for i in range(0,dim):

                if self.data[i][-1]*(np.dot(self.w,self.data[i][:-1]) + self.b) <= 0:

                    self.w = self.w + (self.n * self.data[i][-1] * self.data[i][:-1])
                    self.b = self.b + (self.n * self.data[i][-1]*(self.R**2))
                    self.k += 1
        """
        print(self.b)
        print(self.w)
        print(self.k)
        print('done')
         """       
    def predict(self,new_point):

        pred = (np.dot(self.w,new_point) + self.b)
        if pred > 0:
            return 1.
        elif pred < 0:
            return -1.
        else:
            return 0.



df = pd.read_csv('breast-cancer-wisconsin.data')
df.drop(['id'],1, inplace=True)

#did it 'couse the Bare_Nuclei colum was object type, not numeric, so pandas wasnt showing the 
#the mean() of this column 
#df['Bare_Nuclei'] = df.Bare_Nuclei.convert_objects(convert_numeric=True) or
df['Bare_Nuclei'] = pd.to_numeric(df.Bare_Nuclei,downcast='integer', errors='coerce')

df.fillna(df.mean(),inplace=True)


### this part could be done using sklearn ###

#set all to values as floats
full_data = df.astype(float).values.tolist()

#shuffle the data
random.shuffle(full_data) 

for i in range(0,len(full_data)):
    if full_data[i][-1] == 2:
        full_data[i][-1] = 1
    elif full_data[i][-1] == 4:
        full_data[i][-1] = -1


test_size = 0.2

train_data = np.array(full_data[:-int(test_size*len(full_data))])
test_data = np.array(full_data[-int(test_size*len(full_data)):])
#########


##### test the alg in some real data####

soma = 0
for time in range(0,10000):

    total = 0
    correct = 0
    bound = 0

    a = perceptron()
    a.training(train_data,9)

    for test in test_data:
        
        if a.predict(test[:-1]) == test[-1]:
            correct += 1 
        elif a.predict(test[:-1]) == 0:
            bound += 1
        
        total += 1

    #print(correct,bound,total,correct/total)
    soma += correct/total

print(soma/10000)
################



def plot_data(data):
    for point in data:
        if point[-1] == 1:
            plt.scatter(point[0],point[3], color='b')
        else:
            plt.scatter(point[0],point[3], color='r')
    plt.show()