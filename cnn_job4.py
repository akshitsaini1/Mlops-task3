import os
from keras.layers import Dense, Conv2D , MaxPooling2D ,AveragePooling2D , Flatten
from keras.models import Sequential , load_model , Model
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam , RMSprop
from keras.layers.normalization import BatchNormalization

task3='/task3'
dataset='dataset/cnn'
dataset_path=os.path.join(task3,dataset)
job2='job3'
job3='job3'
job4='job4'
def loss_finder():#used in compilationfind the loss from prev model
    prev_model=load_model(os.path.join(task3,job3,'cnn_model.hdf5'))
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


def get_inter_model(files):
    print('get_init')
    ac_time=0
    fil=''
    for i in files:
        print(i)
        if i.startswith('tunni') and i.endswith('.hdf5'):
            if ac_time < os.path.getmtime(os.path.join('/task3/job4/cnn_intermidiatemodels',i)):
                ac_time=os.path.getmtime(os.path.join('/task3/job4/cnn_intermidiatemodels',i))
                fil=i
    print(fil)
    print(os.path.join('/task3/job4/cnn_intermidiatemodels',fil))
    return os.path.join('/task3/job4/cnn_intermidiatemodels',fil)


def starting_epoch():
    files=os.listdir('/task3/job4/cnn_intermidiatemodels')
    fil=get_inter_model(files)
    start=fil.find('eg')+2
    end=fil.find('.')
    initial_epoch=int(fil[start:end])
    return initial_epoch


def model_load():
    print("loading model...")
    print("model dir= ",os.path.join(task3,job3,'cnn_model.hdf5'))
    my_model=load_model(os.path.join(task3,job3,'cnn_model.hdf5'))
    print("fetching accuracy...")
    acc_file=open(os.path.join(task3,job3,'cnn_accuracy.txt'))
    accuracy=float(acc_file.read().split()[0])
    acc_file.close()
    loss=loss_finder()
    train_model(my_model,accuracy,False,loss)

#if the docker has completed the training model or not
def init_model():##1st function which will run wherenver this file runs
    #check if intermidiate file dir exists in job4
    if not os.path.isdir(os.path.join(task3,job4,'cnn_intermidiatemodels')):
        print("No intermidiate dir found... creating intermidiate dir at /task3/job4/cnn_intermidiatemodels")
        os.mkdir('/task3/job4/cnn_intermidiatemodels')
        model_load()
    
    else:
        files=os.listdir('/task3/job4/cnn_intermidiatemodels')
        if len(files) > 0:#contains intermidiate saved models 
            resume_training(files)
        #when there is no inter model then strart 1st training
        else:
            model_load()
                    
                    
def resume_training(files):
    if not os.path.isfile('/task3/job4/cnn_intermidiatemodels/cnn_twerkedaccuracy.txt'):#the model was stoped b-b/w the trainingi
        print("Training resuming...")
        acc_file=open(os.path.join(task3,job3,'cnn_accuracy.txt'))
        accuracy=float(acc_file.read().split()[0])
        print("files=",files)
        model=get_inter_model(files)#fetch the recent saved intermidaite model
        print(model)
        model=load_model(model)
        loss=loss_finder()
        train_model(model,accuracy,True,loss)
    else:
        print("resuming\n files= ",files)
        model=get_inter_model(files)
        end(model)
        
        
def train_test(path,class_mode):#divide train and testing data and called by train_model only
    training_dataset=os.path.join(path,'training_set')
    test_dataset=os.path.join(path,'test_set')
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory(
        training_dataset,
        target_size=(64, 64),
        batch_size=32,
        class_mode=class_mode)
    test_set = test_datagen.flow_from_directory(
        test_dataset,
        target_size=(64, 64),
        batch_size=32,
        class_mode=class_mode)
    return training_set, test_set


def train_model(model,accuracy,resume,losses):
    checkpoint=ModelCheckpoint('/task3/job4/cnn_intermidiatemodels/tunning_eg{epoch:02d}.hdf5',period=1,verbose=1,monitor='loss')
    print("checkpoint made")
    class_mode=''
    print('loss= ',losses)
    #determine the class_mode
    if losses == 'binary_crossentropy':
        class_mode= 'binary'
    elif losses == 'categorical_crossentropy':
        class_mode = 'categorical'
    print('class_mode= ',class_mode)
    #divide the dataset
    print('dataset= ',dataset_path)
    epoch=2
    train_set,test_set=train_test(dataset_path,class_mode)
    #check if model has to resume the training or start tweaking from start
    if resume: #resume the model iff container stopped during fine_tunning
        if losses == 'binary_crossentropy':
            model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy', metrics=['accuracy'])
        
        init_epoch=starting_epoch()   	

        model.fit(train_set,
        initial_epoch=init_epoch,
        steps_per_epoch=8000/32,#TO MAKE IT FAST REDUCE IT
        epochs=3,
        validation_data=test_set, 
        callbacks=[checkpoint],verbose=1,
        validation_steps=800)
        
        accuracy=model.history.history['accuracy'][0]
        print(accuracy)
        write_accuracy(str(accuracy*100))
        end(model)
    
    else: #run doing fine tunning from starting and check how many time finine tunning has already done
        tweaking(model,accuracy,losses,checkpoint,train_set,test_set)

        
def counter():
    f=open(os.path.join(task3,job4,'counter.txt'))
    count=int(f.read().split()[0])
    f.close()
    count=int(count)
    return count

def find_flatten(model):
    i=0
    j=-1
    for layer in model.layers:
        if layer.__class__.__module__.split('.')[2] == 'pooling':
            j=i;
        i+=1
    i=len(model.layers)
    i-j
    return j+1

def fine_tuning(model,loss,input_shape,last_unit):
    if loss == 'binary_crossentropy':
        last_activation='sigmoid'
    else:
        last_activation='softmax'
        
    top_model=model.output
    unit=input_shape[0]*input_shape[1]*input_shape[2]
    
    in_shape=input_shape[0]
    #it'll keep adding convo and poolinf layer until pooling o/p is < 7
    i=0
    while in_shape > 7:
        top_model=Conv2D(filters=input_shape[0],kernel_size=(2,2),strides=(1,1),name='new_conv'+str(i))(top_model)
        top_model= AveragePooling2D(pool_size=(2, 2),strides=(1,1))(top_model)
        in_shape= in_shape/2
        i+=1
        
    top_model=Flatten()(top_model)
    top_model=Dense(units=int(unit/10),activation='relu')(top_model)
    top_model=Dense(units=int(unit/20),activation='relu')(top_model)
    print(last_unit)
    top_model=Dense(units=last_unit,activation=last_activation)(top_model)
    model=Model(inputs=model.input, outputs=top_model)
    return model
    

#train the model acc to the counter
def tweaking(model,accuracy,loss,checkpoint,train_set,test_set):
    
    for layer in model.layers:#make all the layers trainable
        layer.trainable=True
    count=counter()#count the numbers of times job4 it running 
    if count == 1:
        flat_index=find_flatten(model)
        prev_layer=model.layers[flat_index:]##store the layers info of prev model from flat to the last
        input_shape=model.layers[flat_index-1].output_shape[1:]#output shape of pooling layer is input of next layer
        print("flat_index= ",flat_index)
        print(input_shape)
        
        if input_shape[0] < 7:
            print("no further convo layer can be add... tweaking by changing hyperparameters ")
            lr=0.0001
            if accuracy < 75.00:
                epoch=22
            else:
                epoch=75
        
            model.compile(optimizer=Adam(learning_rate=lr),loss=loss, metrics=['accuracy'])
            model.fit(train_set,
                  epochs=10,
                  steps_per_epoch=8000,
                  validation_data=test_set,
                    callbacks=[checkpoint],
			verbose=1,
                   validation_steps=800)
            
            accuracy=model.history.history['accuracy'][0]
            write_accuracy(str(accuracy*100))
            end(model)
        else:
            print("tweaking by adding some convo layers...")
            total_len=len(model.layers)
            k=flat_index
            print('removing fc layers')
            while k != total_len:            
                model.pop()
                k+=1
            print("layers removed successfully")   
            #fine tunning
            if os.path.isdir(os.path.join(dataset_path,'test_set')):
                print(True)
            else:
                print(False)
            num_files=(os.listdir(os.path.join(dataset_path,'test_set')))#list the number of classes
            print(num_files)
            for f in num_files:
                if f.startswith('.'):
                    num_files.remove(f)
            print(len(num_files))
            model=fine_tuning(model,loss,input_shape,len(num_files)-1)
            
            model.compile(optimizer=Adam(learning_rate=0.0001),loss=loss, metrics=['accuracy'])
            model.fit(train_set,
            steps_per_epoch=8000,
            epochs=10,
            validation_data=test_set,
            callbacks=[checkpoint],verbose=1,
            validation_steps=800)
        
            accuracy=model.history.history['accuracy'][0]
            write_accuracy(str(accuracy*100))
            end(model)          
def end(model):
    model.save('/task3/job3/cnn_model.hdf5')
    twerk_f=open('/task3/job4/cnn_intermidiatemodels/cnn_twerkedaccuracy.txt')
    acc=float(twerk_f.read().split()[0])
    print("new accuracy= ",acc)
    acc_file=open(os.path.join(task3,job3,'cnn_accuracy.txt'),mode='w')
    acc_file.write(str(acc))
    acc_file.close()
    twerk_f.close()
    files=os.listdir('/task3/job4/cnn_intermidiatemodels')
    print("removing intermidiate models and files...")
    for f in files:
        os.remove(f)
    print("files removed.")
        
def write_accuracy(accuracy):
    f=open('/task3/job4/cnn_intermidiatemodels/cnn_twerkedaccuracy.txt','w')
    f.write(accuracy)
    f.close()
    
    
    
init_model()
