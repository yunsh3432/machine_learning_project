
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


def oversampling(example):
    tempexample = example
    value = example[:,[-1]]
    j = 0
    for i in value:      
        if i == 9.:
            for k in range(2501):
                tempexample = np.append(tempexample,[example[j]],axis = 0)
            j = j+1    
        elif i == 8.:
            for k in range(2501):
                tempexample = np.append(tempexample,[example[j]],axis = 0)
            j = j+1   
        elif i == 7.:
            for k in range(2001):
                tempexample = np.append(tempexample,[example[j]],axis = 0)
            j = j+1
        elif i == 6.:
            for k in range(351):
                tempexample = np.append(tempexample,[example[j]],axis = 0)
            j = j+1
        elif i == 5.:
            for k in range(211):
                tempexample = np.append(tempexample,[example[j]],axis = 0)
            j = j+1
        elif i == 4.:
            for k in range(131): 
                tempexample = np.append(tempexample,[example[j]],axis = 0)
            j = j+1
        elif i == 3.:
            for k in range(25):
                tempexample = np.append(tempexample,[example[j]],axis = 0)
            j = j+1
        elif i == 2.:
            for k in range(11):
                tempexample = np.append(tempexample,[example[j]],axis = 0)
            j = j+1
        else:
            pass  
    return tempexample     


# In[3]:


xy_train= np.loadtxt('poker_hand_training.csv',delimiter = ',',dtype = np.float32)
xy_test =  np.loadtxt('poker_hand_test.csv',delimiter = ',',dtype = np.float32)

overxy_train = oversampling(xy_train)
#overxy_train = xy_train

x_data = overxy_train[:,0:-1]
y_data = overxy_train[:,[-1]]

x_test = xy_test[:0:-1]
y_test = xy_test[:,[-1]]
nb_classes = 10 #0~9

X = tf.placeholder(tf.float32,[None,10])
Y = tf.placeholder(tf.int32,[None,1])
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) 


# In[4]:


#varialble
learning_rate = 0.001
learning_epochs = 15
batch_size = 100


# In[5]:


W1 = tf.get_variable("W1", shape=[10,16],
          initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([16]),name = 'bias1')
layer1 = tf.nn.relu(tf.matmul(X,W1)+b1)

W2 = tf.get_variable("W2", shape=[16,32],
         initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([32]),name = 'bias2')
layer2 = tf.nn.relu(tf.matmul(layer1,W2)+b2)
    
W3 = tf.get_variable("W3", shape=[32,64],
                initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([64]),name = 'bias3')
layer3 = tf.nn.relu(tf.matmul(layer2,W3)+b3)
    
W4 = tf.get_variable("W4", shape=[64,128],
                initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([128]),name = 'bias4')
layer4 = tf.nn.relu(tf.matmul(layer3,W4)+b4)
    
W5 = tf.get_variable("W5", shape=[128,128],
                initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([128]),name = 'bias5')
layer5 = tf.nn.relu(tf.matmul(layer4,W5)+b5)
            
W6 = tf.get_variable("W6", shape=[128,64],
                initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([64]),name = 'bias6')
layer6 = tf.nn.relu(tf.matmul(layer5,W6)+b6)

W7 = tf.get_variable("W7", shape=[64,64],
                initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([64]),name = 'bias7')
layer7 = tf.nn.relu(tf.matmul(layer6,W7)+b7)

W8 = tf.get_variable("W8", shape=[64,64],
                initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([64]),name = 'bias8')
layer8 = tf.nn.relu(tf.matmul(layer7,W8)+b8)

W9 = tf.get_variable("W9", shape=[64,32],
                initializer=tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.random_normal([32]),name = 'bias9')
layer9 = tf.nn.relu(tf.matmul(layer8,W9)+b9)

W10 = tf.get_variable("W10", shape=[32,32],
                initializer=tf.contrib.layers.xavier_initializer())
b10 = tf.Variable(tf.random_normal([32]),name = 'bias10')
layer10 = tf.nn.relu(tf.matmul(layer9,W10)+b10)


W11 = tf.get_variable("W11", shape=[32,32],
                initializer=tf.contrib.layers.xavier_initializer())
b11 = tf.Variable(tf.random_normal([32]),name = 'bias11')
layer11 = tf.nn.relu(tf.matmul(layer10,W11)+b11)


W12 = tf.get_variable("W12", shape=[32,64],
                initializer=tf.contrib.layers.xavier_initializer())
b12 = tf.Variable(tf.random_normal([64]),name = 'bias12')
layer12 = tf.nn.relu(tf.matmul(layer11,W12)+b12)


W13 = tf.get_variable("W13", shape=[64,64],
                initializer=tf.contrib.layers.xavier_initializer())
b13 = tf.Variable(tf.random_normal([64]),name = 'bias13')
layer13 = tf.nn.relu(tf.matmul(layer12,W13)+b13)


W14 = tf.get_variable("W14", shape=[64,64],
                initializer=tf.contrib.layers.xavier_initializer())
b14 = tf.Variable(tf.random_normal([64]),name = 'bias14')
layer14 = tf.nn.relu(tf.matmul(layer13,W14)+b14)


W15 = tf.get_variable("W15", shape=[64,32],
                initializer=tf.contrib.layers.xavier_initializer())
b15 = tf.Variable(tf.random_normal([32]),name = 'bias15')
layer15 = tf.nn.relu(tf.matmul(layer14,W15)+b15)


W16 = tf.get_variable("W16", shape=[32,32],
                initializer=tf.contrib.layers.xavier_initializer())
b16 = tf.Variable(tf.random_normal([32]),name = 'bias16')
layer16 = tf.nn.relu(tf.matmul(layer15,W16)+b16)


W17 = tf.get_variable("W17", shape=[32,32],
                initializer=tf.contrib.layers.xavier_initializer())
b17 = tf.Variable(tf.random_normal([32]),name = 'bias17')
layer17 = tf.nn.relu(tf.matmul(layer16,W17)+b17)


W18 = tf.get_variable("W18", shape=[32,10],
                initializer=tf.contrib.layers.xavier_initializer())
b18 = tf.Variable(tf.random_normal([10]),name = 'bias18')
layer18 = tf.nn.relu(tf.matmul(layer17,W18)+b18)


W19 = tf.get_variable("W19", shape=[10,10],
                initializer=tf.contrib.layers.xavier_initializer())
b19 = tf.Variable(tf.random_normal([10]),name = 'bias19')
layer19 = tf.nn.relu(tf.matmul(layer18,W19)+b19)

W20 = tf.get_variable("W20", shape=[10,nb_classes],
                initializer=tf.contrib.layers.xavier_initializer())
b20 = tf.Variable(tf.random_normal([nb_classes]),name = 'bias20')
hypothesis = tf.matmul(layer19,W20)+b20
     


# In[6]:


#optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
logits = hypothesis,labels = Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[7]:


#training output
predicted = tf.argmax(hypothesis ,1)
correnct_prediction = tf.equal(predicted,tf.argmax(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correnct_prediction,tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(learning_epochs):
    avg_cost = 0
    for i in range(1000):
        batch_xs = x_data
        batch_ys = y_data#mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ ,acc= sess.run([cost, optimizer,accuracy], feed_dict=feed_dict)
        avg_cost += c / 1000

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print('Acc: ', acc)
print('Learning Finished!')


# In[ ]:


#test output

