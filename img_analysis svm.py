#Pour la colorness : dim_zone  90 / 100 sont des bonnes dimensions pour étudier
#Tester le contraste / la variance entre les pixels de la zone : probablement une dim plus petite
import os
from typing import List, Tuple
import random
from collections import Counter
from sklearn import metrics
import numpy as np
from sklearn import svm
from collections import defaultdict



from PIL import Image

current_path = os.getcwd()
#Path folder containing the folders images and images_irc
os.chdir(current_path+"/ressources/")

#For the folder 100images:
classification = ["Beton","Ardoises","Zinc","Tuiles","Ardoises","Ardoises","Tuiles","Tuiles","Béton","Tuiles","Tuiles","Ardoises","Tuiles","Tuiles","Ardoises","Tuiles","Ardoises","Beton","Ardoises","Beton","Beton","Zinc","Ardoises","Tuiles","Ardoises","Ardoises","Ardoises","Ardoises","Zinc","Ardoises","Zinc","Ardoises","Beton","Zinc","Tuiles","Beton","Tuiles","Tuiles","Beton","Beton","Tuiles","Tuiles","Tuiles","Tuiles","Ardoises","Ardoises","Ardoises","Tuiles","Ardoises","Tuiles","Tuiles","Zinc","Tuiles","Ardoises","Tuiles","Ardoises","Beton","Tuiles","Zinc","Tuiles","Beton","Ardoises","Tuiles","Ardoises","Tuiles","Ardoises","Tuiles","Ardoises","Ardoises","Ardoises","Tuiles","Ardoises","Tuiles","Tuiles","Ardoises","Tuiles","Tuiles","Ardoises","Ardoises","Zinc","Tuiles","Ardoises","Beton","Ardoises","Tuiles","Ardoises","Tuiles","Tuiles","Tuiles","Tuiles","Tuiles","Ardoises","Tuiles","Ardoises","Ardoises","Ardoises","Ardoises","Tuiles","Tuiles","Tuiles"]

dim_zone = 50
eps = 1e-10

#Function that sort a set with frequencies
def create_set_with_frequencies(lst):
    frequency_dict = {}
    for item in lst:
        if item in frequency_dict:
            frequency_dict[item] += 1
        else:
            frequency_dict[item] = 1

    frequency_set = set()
    for item, count in frequency_dict.items():
        frequency_set.add((item, count))

    sorted_list = sorted(list(frequency_set), key=lambda x: x[1], reverse=True)
    return sorted_list

#Main function that return the "colorness" of an image (aka sum of value of each blue pixel of the image divide by sum of value of each blue pixel of the image with the most blue)
def colorness(numbers):
    res = []
    for i in numbers[:int(len(numbers)-100)]:
        # Open the image
        im = Image.open('images/'+str(i[0])+'.jpg')
        image = im.crop((128-(dim_zone/2), 128-(dim_zone/2), 128+(dim_zone/2), 128+(dim_zone/2)))

        width, height = image.size

        # Initialize variables to store the total red, green, and blue values
        total_red = 0
        total_green = 0
        total_blue = 0
        total = 0

        # Iterate over the pixels in the image
        for x in range(width):
            for y in range(height):
                # Get the red, green, and blue values of the pixel
                r, g, b = image.getpixel((x, y))
                # Add the values to the totals
                total_red += r
                total_green += g
                total_blue += b
                total += r +g +b


        prc_red = total_red / total
        prc_green = total_green / total
        prc_blue = total_blue / total

        res.append((i,prc_red,prc_green,prc_blue))

    min_red = min(res, key=lambda x: x[1])[1]
    min_green = min(res, key=lambda x: x[2])[2]
    min_blue = min(res, key=lambda x: x[3])[3]
    mini = min(min_red,min_green,min_blue)

    max_red = max(res, key=lambda x: x[1])[1]
    max_green = max(res, key=lambda x: x[2])[2]
    max_blue = max(res, key=lambda x: x[3])[3]
    maxi = max(max_red,max_green,max_blue)

    res2 = []
    for i in range(len(res)):
        res2.append([res[i][0]])
        for j in range(1,4):

            res2[i].append((float(res[i][j]) - mini)/(maxi-mini))


    return (mini,maxi,res2)

# Same than colorness but only for 1 specific images
def colorness_solo(file,mini,maxi):

    # Open the image
    im = Image.open("images/"+'/'+str(file[0])+".jpg")
    image = im.crop((128-(dim_zone/2), 128-(dim_zone/2), 128+(dim_zone/2), 128+(dim_zone/2)))

    width, height = image.size

    # Initialize variables to store the total red, green, and blue values
    total_red = 0
    total_green = 0
    total_blue = 0
    total = 0

    # Iterate over the pixels in the image
    for x in range(width):
        for y in range(height):
            # Get the red, green, and blue values of the pixel
            r, g, b = image.getpixel((x, y))
            # Add the values to the totals
            total_red += r
            total_green += g
            total_blue += b
            total += r +g +b


    prc_red = total_red / total
    prc_green = total_green / total
    prc_blue = total_blue / total

    return(file,(prc_red-mini)/(maxi-mini),(prc_green-mini)/(maxi-mini),(prc_blue-mini)/(maxi-mini))

#Return list sorted using knn
def k_nearest_neighbors(k: int, data: List[Tuple[int,int, int, int]], point: Tuple[int,int, int, int]) -> List[Tuple[int, int, int, int]]:
    # Calculate the Euclidean distance between the point and each element in the data
    distances = [(x[1] - point[1]) ** 2 + (x[3] - point[3]) ** 2 + (x[2] - point[2]) ** 2 for x in data]

    # Sort the data by distance
    sorted_data = [x for _, x in sorted(zip(distances, data))]

    # Return the k nearest neighbors
    return sorted_data[:k]

#Return list of material manually attribute to each image
def getResult():
    with open('images/batiments.csv', 'r') as f:

        # Create an empty list
        materiau = []
        cpt=0

        # Iterate over the rows in the CSV file
        for row in f:
            row = row.split(';')
            # Append the second element of the row to the list
            materiau.append([cpt,row[2].split('\n')[0]])
            cpt+=1
    return materiau

def calculate_weighted_f1_score(confusion_matrix, weights):
    # calculate precision for each class
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis = 0)
    # calculate recall for each class
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis = 0)
    # calculate F1-score for each class
    f1_score = 2 * (precision * recall) / (precision + recall)
    # weight the F1-score for each class
    f1_score = np.nan_to_num(f1_score)
    #print(f1_score)
    weighted_f1_score = f1_score * weights
    # return the average weighted F1-score
    return np.sum(weighted_f1_score)

#Initialization
choice = []
result_1 = getResult()

curves = []

X=[]
Y=[]

erreur=[]
bonnes=[]
print("debut")
test_size=300 #Size of test set

#Start the process of testing the score of our programm
for p in range(1000,1200,400) :

    pop=p #pop : size of train set

    cpt=0
    #Test 5 differents set to get average sucess %
    for i in range(5):
        random.shuffle(result_1)
        mini,maxi,color = colorness(result_1) #Get colorness for each image
        #Divide images into train set and test set (aka result and test)
        print(color)
        result=color[:pop]
        test=color[len(color)-test_size:]
        label=[] #Get real materail value for each image in train set
        reeltmp = []
        test_label=[] #Get real materail value for each image in test set

        verif=[]
        for e in test:
            verif.append(e[0])

        #Zinc,Tuiles,Beton,Ardoises
        for e in result :
            if e[0][1]=='Zinc Aluminium':
                label.append(0)
            if e[0][1]=='Tuiles':
                label.append(1)
            if e[0][1]=='Beton':
                label.append(2)
            if e[0][1]=='Ardoises':
                label.append(3)
            e.pop(0)

        for e in test :
            if e[0][1]=='Zinc Aluminium':
                test_label.append(0)
            if e[0][1]=='Tuiles':
                test_label.append(1)
            if e[0][1]=='Beton':
                test_label.append(2)
            if e[0][1]=='Ardoises':
                test_label.append(3)
            e.pop(0)

        # Create SVM Model
        clf = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')

        color2=np.array(result)


        for h in result_1:
            if h[1]=="Zinc":
                reeltmp.append(0)
            if h[1]=="Tuiles":
                reeltmp.append(1)
            if h[1]=="Beton":
                reeltmp.append(2)
            if h[1]=="Ardoises":
                reeltmp.append(3)


        for h in range(len(test_label)):
            if(reeltmp[h]!=test_label[h]):
                erreur.append(result_1[h][0])
        for h in range(len(test_label)):
            if(reeltmp[h]==test_label[h]):
                bonnes.append(result_1[h][0])


        clf.fit(color2,label)
        prediction = clf.predict(test)

        for h in range(len(prediction)):
            if prediction[h]==0:
                verif[h].append("Zinc")
            if prediction[h]==1:
                verif[h].append("Tuiles")
            if prediction[h]==2:
                verif[h].append("Béton")
            if prediction[h]==3:
                verif[h].append("Ardoises")

        #print(verif) #Check give real value and determined value for each image

        # Initialize a counter for common elements
        common = 0

        # Iterate over both lists and compare the corresponding elements
        for a, b in zip(prediction, test_label):
            if a == b:
                common += 1

        # Calculate the similarity percentage
        similarity = common / len(prediction)
        print(similarity)
        cpt +=similarity


    X.append(pop)
    Y.append(cpt*20)

    print(f"Similarity: {cpt * 20}%")

    print(X)
    print(Y)

print(create_set_with_frequencies(erreur))
print(create_set_with_frequencies(bonnes))