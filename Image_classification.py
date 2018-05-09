
# coding: utf-8

# ## Objective : use Neural networks to classify images. In this case we obtained a dataset containing cat and dog images and apply the keras NN API to classify dog and cat images.

# In[4]:


# Include all packages : 
# cv2 is the Image processing library - Can be installed using : sudo pip install --ignore-installed  opencv-python
import os
import pandas as pd
import cv2  
import numpy as np
from sklearn.cross_validation import train_test_split
     
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Preprocessing the images

# #### Group all images info in a single dataframe  and find out width, height and labels of the images

# In[9]:


widths =[]
labels = []
ratios = []
heights = []
full_paths = []


#  list of  images
cwd = os.getcwd()
images_dir = os.path.join(cwd, '/Users/doyelm/Documents/Proj 2/train/') 
images_filenames = os.listdir(images_dir)

# Add each image info to dataframe
for filename in images_filenames:
     if (filename[:3]== 'cat') or (filename[:3]== 'dog'):  
            
        path = images_dir + filename
        img = Image.open(path,'r')
        full_paths.append(images_dir + filename)

        size_im =img.size
        widths.append(size_im[0])
        heights.append(size_im[1])
        ratios.append(float(size_im[0])/float(size_im[1]))
        
        labels.append(['cat', 'dog'].index(filename[:3])) 
        
 
 
images_df = pd.DataFrame({'label': labels,
                          'full_path': full_paths , 
                          'width':widths, 
                          'height':heights, 
                          'ratio':ratios})


# In[10]:


images_df.head()


# In[11]:


images_df.info()


# #### Check if dataset is balanced

# In[12]:


images_df.groupby('label').count()


# ### Reducing Images to gray scale

# ### RGB

# In[14]:


img = mpimg.imread(images_df['full_path'][0])
image = img
plt.subplot(2,3,1) 
plt.imshow(image[:,:,0],cmap='gray')
plt.subplot(2,3,2) 
plt.imshow(image[:,:,1]  , cmap='gray')
plt.subplot(2,3,3) 
plt.imshow(image[:,:,2]  , cmap='gray') 


# ### HSL transform
# 

# In[15]:


image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
plt.subplot(2,3,1) 
plt.imshow(image[:,:,0]  , cmap='gray')
plt.subplot(2,3,2) 
plt.imshow(image[:,:,1]  , cmap='gray')
plt.subplot(2,3,3) 
plt.imshow(image[:,:,2]  , cmap='gray') 


# In[16]:


plt.hist(image.reshape( img.shape[0]*img.shape[1]*img.shape[2]))


# ### HSL normalize

# In[17]:


image = (image*(0.8)/255)-0.4
plt.subplot(2,3,1) 
plt.imshow(image[:,:,0]  , cmap='gray')
plt.subplot(2,3,2) 
plt.imshow(image[:,:,1]  , cmap='gray')
plt.subplot(2,3,3) 
plt.imshow(image[:,:,2]  , cmap='gray')


# In[18]:


plt.hist(image.reshape( img.shape[0]*img.shape[1]*img.shape[2]))


# ### Resizing the images

# In[52]:


img = mpimg.imread(images_df['full_path'][0])
plt.imshow(img)


# In[25]:


img.shape


# In[26]:


img=cv2.resize(img, (75, 75) )


# In[27]:


img.shape


# In[28]:


plt.imshow(img)


# In[30]:


img=cv2.resize(img, (150, 150) )
plt.imshow(img)


# In[19]:


plt.hist(images_df['width'],bins=20)
images_df['width'].describe()
plt.xlabel('width')
plt.ylabel('count')
plt.show()


# In[20]:


plt.hist(images_df['height'],bins=20)
images_df['height'].describe()
plt.xlabel('height')
plt.ylabel('count')
plt.show()


# ### Regenerating all images with optimal size

# In[38]:


#Regenerating the images with the optimal size
Image_hight=150
Image_width=150

def BatchGenerator(df):
    # A function that will use python generator to read the data. 
    # Aventually I did not had to cut the data since the machine I am using is strong enough. 
    
    X = []
    y = []
    # Open the file
    i = 0
    for index, row in df.iterrows():
        img = mpimg.imread(row['full_path']) 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        img = img[:,:,1]
        img = (img*(0.8)/255)-0.4
        
        img=cv2.resize(img, (Image_hight, Image_width) )
        img = img.reshape((Image_hight*Image_width))
        i += 1
        if(i % 2500 == 0):
                print('Batch' )
                yield np.asarray(X), np.asarray(y)
                X = []
                y = [] 
            
        label =  (row['label'] )
        #  print(row['full_path'], steering_angle, img.size, row['height'], row['width'])
        # multiply images with stearing angle !=0

        X.append(img)
        y.append(label)
    
    yield np.asarray(X), np.asarray(y)


# ### preparing the model

# In[42]:



from keras.models import Sequential
 
from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation, Input, ELU
from keras.layers.normalization import BatchNormalization

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
 
model = Sequential()
#model.add(BatchNormalization())

model.add(Conv2D(32, 5, 5, input_shape=(Image_hight, Image_width ,1), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32, 3, 3 , activation='relu'))
model.add(MaxPooling2D((2,2)))
 
model.add(Conv2D(64, 3, 3 , activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
 
model.summary() 

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

for X, y in BatchGenerator(images_df.sample(frac=1, random_state= 1).reset_index(drop=True)[1:25000]):
    X_train, X_val, y_train, y_val = train_test_split(X, y , random_state=5, test_size=0.2)

    history = model.fit(X_train.reshape(X_train.shape[0],
                                        Image_hight, 
                                        Image_width,1), 
                        y_train,
                        batch_size=64, 
                        nb_epoch=5,
                        verbose=1, 
                        validation_data=(X_val.reshape(X_val.shape[0],
                                                       Image_hight, 
                                                       Image_width,1) ,
                                         y_val))
    
    


# In[55]:


model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')


# ### Prediction

# In[57]:


# take the first 3 figures

for X, y in BatchGenerator(images_df.sample(frac=1, random_state= 1).reset_index(drop=True)[0:3]):
     X, y 

#save a single figure into the variable a and plot a, 

a=X[1,:].reshape( Image_hight, Image_width )

plt.imshow(a , cmap='gray' )


#Prediction: 

prediction = model.predict_classes(a.reshape(1,Image_hight, Image_width,1 ))
print(prediction)

#probabuility of the prediction

prediction_proba= model.predict(a.reshape(1,Image_hight, Image_width,1 ))
print( prediction_proba)


# ### The model correctly predicted the image as dog with 0.99 probability
