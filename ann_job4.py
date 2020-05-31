import os
from keras.layers import Dense
from keras.models import Sequential , load_model , Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam , RMSprop
import pandas as pd

task3='/task3'
dataset='dataset/ann'
dataset_path=os.path.join(task3,dataset)
job2='job3'
job3='job3'
job4='job4'

def loss_finder():#used in compilationfind the loss from prev model
    prev_model=load_model(os.path.join(task3,job3,'ann_model.hdf5'))
    if prev_model.layers[-1].__class__.__name__ == 'Activation':
        activation=prev_model.layers[-1].get_config().get('activation')
        print('loss function= ',activation)
        if activation == 'softmax':
            return 'categorical_crossentropy'
        elif activation == 'sigmoid':
            return 'binary_crossentropy'
    elif prev_model.layers[-1].__class__.__name__ == 'Dense':
        activation=prev_model.layers[-1].get_config().get('activation')
        print('loss function= ',activation)
        if activation == 'softmax':
            return 'categorical_crossentropy'
        elif activation == 'sigmoid':
            return 'binary_crossentropy'
        elif activation =='mean_squared_error':
            return 'linear'
        
def get_inter_model(files):
    ac_time=0
    file=''
    for i in files:
        if i.startswith('ann_fine_tunning_eg') and i.endswith('.hdf5'):
            if ac_time < os.path.getmtime(i):
                ac_time=os.path.getmtime(i)
                file=i
    return file


def starting_epoch():
    files=os.listdir('/task3/job4/ann_intermidiatemodels')
    file=get_inter_model(files)
    start=file.find('eg')+2
    end=file.find('.')
    initial_epoch=int(file[start:end])
    return initial_epoch


def load_dataset():
    #load train and test dataset
    train_set=pd.read_csv(os.path.join(dataset_path,'train.csv'))
    test_set=pd.read_csv(os.path.join(dataset_path,'test.csv'))
    
    y_file=open(os.path.join(dataset_path,'y_file.txt'))
    y_columns=y_file.read().split()[0]
    
    y_train=train_set[y_columns]
    X_train=train_set.drop([y_columns],axis=1)    
    
    y_test=test_set[y_columns]
    X_test=test_set.drop([y_columns],axis=1)
    
    return X_train.values,y_train.values,X_test.values,y_test.values

#if the docker has completed the training model or not
def init_model():#1st function which will run wherenver this file runs
    #check if intermidiate file dir exists in job4
    if not os.path.isdir(os.path.join(task3,job4,'ann_intermidiatemodels')):
        print("No intermidiate dir found... creating intermidiate dir at /task3/job4/ann_intermidiatemodels")
        os.mkdir('/task3/job4/ann_intermidiatemodels')
        model_load()
    
    else:
        files=os.listdir('/task3/job4/ann_intermidiatemodels')
        if len(files) > 0:#contains intermidiate saved models 
            resume_training(files)
        #when there is no inter model then strart 1st training
        else:
            model_load()

def model_load():
    print("loading model...")
    print("model dir= ",os.path.join(task3,job3,'ann_model.hdf5'))
    my_model=load_model(os.path.join(task3,job3,'ann_model.hdf5'))
    print("fetching accuracy...")
    acc_file=open(os.path.join(task3,job3,'ann_accuracy.txt'))
    accuracy=float(acc_file.read().split()[0])
    acc_file.close()
    loss=loss_finder()
    train_model(my_model,accuracy,False,loss)

def train_model(model,accuracy,resume,loss):
    checkpoint=ModelCheckpoint('/task3/job4/cnn_intermidiatemodels/tunning_eg{epoch:02d}.hdf5',period=1,verbose=1,monitor='loss')
    print("checkpoint made")
    #load the dataset
    X_train,y_train,X_test,y_test=load_dataset()
    print('dataset loaded')
    
    #check if model has to resume the training or start tweaking from start
    if resume: #resume the model iff container stopped during fine_tunning
        model.compile(optimizer=Adam(learning_rate=0.0001),loss=loss, metrics=['accuracy'])
        
        model.fit(train_set,
        initial_epoch=starting_epoch(),
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set, 
        callbacks=[checkpoint],verbose=1,
        validation_steps=800)
        
        accuracy=model.history.history['accuracy'][0]
        write_accuracy(str(accuracy))
        end(model)
    
    else: #run doing fine tunning from starting and check how many time finine tunning has already done
        tweaking(model,accuracy,loss,checkpoint,X_train,y_train,X_test,y_test)

    
    
          
def tweaking(model,accuracy,loss,checkpoint,X_train,y_train,X_test,y_test):
           
    input_shape=model.output_shape[1:]#output shape of pooling layer is input of next layer
    lr=0.0001
    if accuracy < 75.00:
        epoch=25
    else:
        epoch=75
    model.compile(optimizer=Adam(learning_rate=lr),loss=loss, metrics=['accuracy'])
    model.fit(X_train,
              y_train,
            epochs=epoch,
               callbacks=[checkpoint],verbose=1,
                  )
    accuracy=model.history.history['accuracy'][0]
    write_accuracy(str(accuracy))
    end(model)
def end(model):
    model.save('/task3/job3/ann_model.hdf5')
    twerk_f=open('ann_twerkedaccuracy')
    acc=float(twerk_f.read().split()[0])
    print("new accuracy= ",acc)
    acc_file=open(os.path.join(task3,job3,'ann_accuracy.txt'),mode='w')
    acc_file.write(acc)
    acc_file.close()
    twerk_f.close()
    print("removing intermidiate models and files...")
    for f in files:
        os.remove(f)
    print("files removed.")
        
def write_accuracy(accuracy):
    f=open('ann_twerkedaccuracy','w')
    f.write(accuracy)
    f.close()
