tf.reset_default_graph()

import tensorflow as tf
#X real data and Z generated data
X = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, 100])

#one hidden layer network for generator and discriminator

def generator(z):
  with tf.variable_scope("generator",reuse=tf.AUTO_REUSE):
    init = tf.contrib.layers.xavier_initializer()  #Returns an initializer performing "Xavier" initialization for weights
    h1 = tf.layers.dense(inputs = z, units =128, activation=tf.nn.relu, kernel_initializer=init, use_bias=True) #units: Integer or Long, dimensionality of the output space.
    out = tf.layers.dense(inputs=h1, units=784, activation=tf.nn.tanh, kernel_initializer=init, use_bias=True)
    
    return out

def discriminator(x):
  with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):  #Variable scope allows you to create new variables and to share already created ones while providing checks to not create or share by accident
    init = tf.contrib.layers.xavier_initializer()
    h1 = tf.layers.dense(inputs = x, units =128, activation=tf.nn.relu, kernel_initializer=init, use_bias=True)
    logits = tf.layers.dense(inputs =h1, units =1, kernel_initializer=init, use_bias=True)
    
    return logits
  
G_sample = generator(Z) #Pass some random noise data to the generator to produce the fake data
logits_real = discriminator(X) #pass that fake data to the discriminator D 
logits_fake = discriminator(G_sample) #pass the real data to the discriminator D seperately


#tf.reduce_mean(x, axis) Computes the mean of elements across dimensions of a tensor.
#tf.nn.sigmoid_cross_entropy_with_logits()A Tensor of the same shape as logits with the componentwise logistic losses.
#tf.ones_like()Creates a tensor with all elements set to 1.
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real),logits=logits_real))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake),logits=logits_fake))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake),logits=logits_fake))
#for Real data X we give 1’s as labels and for fake data Z we give 0’s as labels and we apply cross entropy loss to both the logits to calculate the final D loss.
#at generator G we take the same fake logits but here we give 1’s as labels which is complete opposite for D_loss_fake variable.

D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
G_solver =tf.train.AdamOptimizer(learning_rate=1e-3,beta1=0.5).minimize(G_loss, var_list=G_vars)
D_solver =tf.train.AdamOptimizer(learning_rate=1e-3,beta1=0.5).minimize(D_loss, var_list=D_vars)

# scope problem=I don't get this error with neither of the two TensorFlow versions. Note that code block 3 should only be executed once! If you want to execute the block of the graph creation again, TensorFlow will raise this error, because the graph (with it's layer names) already exists. If you run this notebook block after block there should be absolutely no problem.
#In the case you want to re-run codeblock 3 (for what ever reason) just insert a simple tf.reset_default_graph() at the beginning of the block. This will reset the graph you have already create and though you can create it again.

sess = tf.Session() #A class for running TensorFlow operations.
#A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated

sess.run(tf.global_variables_initializer())

#tf.variables_initializer(var_list, name='init')
#This means that we are implitcitly passing tf.global_variables as a var_list into tf.variables_initializer. If we have not defined any variables before calling tf.global_variables_initializer, var_list is essentially empty.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import Image,display


def sample_Z(r, c):
  return np.random.uniform(-1., 1., size=[r,c])
#Draw samples from a uniform distribution of given size


def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return



samples = sess.run(G_sample, feed_dict={Z: sample_Z(128,100)})

fig = show_images(samples[:16])
plt.show()
print()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

for it in range(50000):
    
    if it % 1000 == 0:
      samples = sess.run(G_sample, feed_dict={Z: sample_Z(128,100)})
      fig = show_images(samples[:16])
      plt.show()
      print()
    x,_= mnist.train.next_batch(128)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: x, Z: sample_Z(128, 100)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(128, 100)})
    
    
    if it % 1000 == 0:
      print('Iter: {}'.format(it))
      print('D_loss: {:.4}'.format(D_loss_curr))
      print('G_loss: {:.4}'.format(G_loss_curr))
      print()