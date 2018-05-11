import json
import sys
import operator
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

import matplotlib.pyplot as plt

start_time = time.time()
file = 'BCR_strings_length_5_occ5000_patient_cluster_percents'
data = json.load(open(file + '.json'))

original_data = json.load(open('patient_info.json'))
original_data = original_data["data"]
sys.stderr.write(str(time.time()-start_time)+'seconds to load data \n')

output_matrix = []
cancer_map = {}
cancer_map['SKCM'] = 0
cancer_map['LUSC'] = 1
cancer_map['READ'] = 2
cancer_map['STAD'] = 3
cancer_map['BRCA'] = 4
cancer_map['BLCA'] = 5
cancer_map['KIRC'] = 6
cancer_map['HNSC'] = 7
cancer_map['PRAD'] = 8
cancer_map['LUAD'] = 9
cancer_map['COAD'] = 10
cancer_map['THCA'] = 11
cancer_map['OV'] = 12

j=0
for patient_BCR_record in original_data:
    output_array = np.zeros(13,)
    output_array[cancer_map[patient_BCR_record['cancer_type']]] =1
    #predict cancer type
    #output_matrix.append(output_array)
    #predict living or dead
    output_val = 0
    if original_data[j]['living'] == 'living':
        output_val = 1
    output_matrix.append(output_val)
    j = j + 1

print(len(original_data))
output_matrix = output_matrix

input_matrix = []

cluster_size = 50
cluster_string = "ward_clustering_" + str(cluster_size)
i=0
for obj in data["data"]:
    cluster_results = obj[cluster_string]
    if(len(obj[cluster_string])>0):
        # if i == 0:
            # print(obj[cluster_string])
            # print(len(obj[cluster_string]))
        input_array = np.zeros(cluster_size + 1,)
        for cluster_label in cluster_results:
            if(cluster_label != '-1' and cluster_label!=-1):
                input_array[int(cluster_label)] = obj[cluster_string][cluster_label]

        #add cancer type to input
        input_array[cluster_size] = cancer_map[original_data[i]['cancer_type']] 
        input_matrix.append(input_array)
        i=i+1


print(input_matrix[0:1])
if len(input_matrix) > 0:
    input_matrix = input_matrix
    
    X_train, X_test, y_train, y_test = train_test_split(input_matrix, output_matrix, test_size=0.2, random_state=23)

    X_train = np.array(X_train)
    y_train = np.array(y_train)


    # K.clear_session()
    model = Sequential()
    model.add(Dense(2000, input_dim=cluster_size+1, activation="relu"))
    model.add(Dropout(0.50))
    model.add(Dense(2000, activation="relu"))
    model.add(Dropout(0.10))
    model.add(Dense(1, activation="sigmoid"))
    Adam = optimizers.adam(lr=0.01)
    model.compile(loss="binary_crossentropy", optimizer=Adam, metrics=["accuracy"])

    # model = Sequential()
    # model.add(Dense(100, input_dim=cluster_size+1, activation="relu"))
    # model.add(Dense(1, activation="softmax"))
    # #model.add(Dense(13, activation="softmax"))
    # print("Dense add 1")
    # #model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # model.compile(loss="hinge", optimizer="adam", metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=20, batch_size=32)
    print("model fit")
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss1.png')

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    scores = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    print(predictions[0:10])

    print("\nAccuracy: %.2f%%" % (scores[1]*100))
elif len(input_matrix) == 0:
    print("No data for cluster approach")
