import multiprocessing
import face_recognition
import cv2
from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from model_structure import All_models
import os
import json
from glob import glob


class generate_encodings():
    knownNames = []
    knownencoding = []
    @staticmethod
    def test(t):
        name = t.split("/")[-2]
        # print(name)
        # exit()
        image = cv2.imread(t)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='cnn')
        encodings = face_recognition.face_encodings(rgb, boxes)
        return [encodings[0], name]

    # generate encodings from images and gives output of pickle file.
    def encodings(self,dataset,filename,name):
        print("Parellel Process Number-->", multiprocessing.cpu_count())
        self.pool = multiprocessing.Pool(multiprocessing.cpu_count())
        self.tasks = []
        # for folder in glob(dataset+"/*"):
        for img in glob(dataset + "/*"):
            self.tasks.append(img)
        self.results = []
        for t in self.tasks:
            self.temp = self.pool.apply_async(self.test, [t])
            self.results.append(self.temp)
        self.knownEncodings=[]
        self.knownNames=[]
        for result in tqdm(self.results):
            check_point = type(result.get()) is list
            if check_point == False:
                continue
            elif check_point == True:
                a, b = result.get()
                self.knownEncodings.append(a)
                self.knownNames.append(b)

        self.pool.close()
        self.pool.join()
        self.final = {"encodings": self.knownEncodings, "names": self.knownNames}
        # print(self.final)
        # exit()
        # current_pickle
        f1 = open("./current_pickle/" + filename + name + ".pickle", "wb")
        f1.write(pickle.dumps(self.final))
        f1.close()
        f = open("./single_pickle/" + filename +name+ ".pickle", "wb")
        f.write(pickle.dumps(self.final))
        f.close()
        return filename +name

class Train():
    # function for generate accuracy vs validation_accuracy graph of model
    def plot_acc(self,history,model_name):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("model_test/" + model_name + "_acc_VS_val-acc" + ".png")
        plt.show()

    # function for generate loss vs validation_loss graph of model
    def plot_val(self,history,model_name):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("model_test/" + model_name + "_loss_VS_val-loss" + ".png")
        plt.show()

    # main function for training
    def start_train(self,model_name,file_name):
        data = pickle.loads(open("single_pickle/"+file_name+".pickle", "rb").read())
        df = pd.DataFrame.from_dict(data)
        # print(df)
        names = df['names'].unique()
        name_dict = {}
        for i, name in enumerate(names):
            name_dict[i] = name
        with open('model_label/'+model_name+".json", 'w') as fp:
            json.dump(name_dict, fp)
        name_dict2 = {y: x for x, y in name_dict.items()}
        y = list(df['names'].replace(name_dict2))
        X = list(df['encodings'])
        # print(y)
        # print(X)
        number_calsses = len(names)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=47)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        model = All_models.Simple_NeuralNet(number_calsses)
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])
        model.summary()
        # model_json = model.to_json()
        # with open("./output/model_architecture/" + model_name + ".json", "w") as json_file:
        #     json_file.write(model_json)
        history = model.fit([X_train], [y_train], epochs=500, batch_size=50, validation_split=0.30)
        model.save("model/" + model_name + ".model")
        loss_and_metrics = model.evaluate([X_test], [y_test], batch_size=50)
        print("loss and metrics", loss_and_metrics)
        # summarize history for accuracy
        self.plot_acc(history,model_name)
        # summarize history for loss
        self.plot_val(history,model_name)
        return model_name

class Merge_encodings():
    # function for Merge all single encodings
    def merge(self,files=[]):
        list_1 = []
        list_2 = []
        for file in tqdm(files):
            data = pickle.loads(open(file, "rb").read())
            # print(data)
            # exit()
            for name in data['names']:
                list_1.append(name)
            for encodes in data['encodings']:
                list_2.append(encodes)
            os.remove(file)
        data = {"encodings": list_2, "names": list_1}
        merge_file_name = str(len(set(data['names'])))+"_labled_merge"
        f1 = open("single_pickle/" + merge_file_name + ".pickle", "wb")
        f1.write(pickle.dumps(data))
        f1.close()

        # f = open("merged_pickle/"+merge_file_name+".pickle", "wb")
        # f.write(pickle.dumps(data))
        # f.close()


        return merge_file_name