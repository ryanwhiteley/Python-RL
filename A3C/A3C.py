import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal

%matplotlib inline
%load_ext tensorboard

import os
os.getcwd()
os.chdir("ReinforcementLearning/A3C")
from helper import *

from random import choice
from time import sleep
from time import time

#%%

max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = 4 # Observation Dim
a_size = 2 # Actions
load_model = False
model_path = './model'

#%%

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
    num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(i,s_size,a_size,trainer,model_path,global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

#%%

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)

#%%
i=0
for e in tf.train.summary_iterator("train_0/events.out.tfevents.1566753642.Ryans-MacBook-Pro.local"):
    for v in e.summary.value:
        i += 1
        if v.tag == "Perf/Reward" and i%100 == 1:
            print("Iteration no. {:d}: {:0.3f}\n".format(i,v.simple_value))

it = tf.train.summary_iterator("train_0/events.out.tfevents.1566753642.Ryans-MacBook-Pro.local")
v = it.__next__()

#%%

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
    i = 0
    env = gym.make("CartPole-v0")
    s = env.reset()
    myAgent = workers[0]

    while i <= 1000:
        env.render()
        a_dist = sess.run(myAgent.local_AC.policy,feed_dict={myAgent.local_AC.inputs:[s]})
        a = np.random.choice(a_dist[0],p=a_dist[0])
        a = np.argmax(a_dist == a)
        s,r,d,_ = env.step(a)
        i += 1
