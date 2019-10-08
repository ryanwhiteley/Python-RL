import numpy as np
import scipy as sp
import gym
from time import sleep
episodes = 50
episode_trials = 5
episode_length = 100
lim = 5

class Model():
    def __init__(self):
        self.bestrewards = 0
        self.bestcontrol = []
        self.rewards = 0
        self.env = gym.make('CartPole-v0')

    def train(self,episodes,episode_trials,episode_length,lim,train_til_good = False):
        print("Training...\n")
        neglim = -lim
        if train_til_good == True:
            episodes2 = 10000
        for i in range(episodes2):
            if i%100 == 0 and train_til_good == False:
                print("Episode {:d} of {:d} complete.\n".format(i,episodes2))
            if train_til_good == True and self.bestrewards >= 490:
                break
            self.rewards = 0
            control = np.random.uniform(low=0,high=lim,size=4)
            for j in range(episode_trials):
                s = self.env.reset()
                d = False
                k = 0
                while d == False and k<episode_length:
                    a = 0 if (np.dot(control,np.array(s))<=0) else 1
                    s,r,d,_ = self.env.step(a)
                    self.rewards += r - np.abs(s[0])
                    k += 1
            if self.rewards > self.bestrewards:
                self.bestrewards = self.rewards
                self.bestcontrol = control
        print("Done!")

    def display(self,length = 1000):
        if self.bestcontrol == []:
            print("Train model first.")
        else:
            s = self.env.reset()
            d = False
            k = 0
            while d == False and k<length:
                self.env.render()
                a = 0 if (np.dot(self.bestcontrol,np.array(s))<=0) else 1
                s,r,d,_ = self.env.step(a)
                k += 1
                sleep(0.05)


agent = Model()
agent.train(episodes=episodes,episode_trials=episode_trials,episode_length=episode_length,lim=lim,train_til_good = True)
agent.display()
a = np.array([4.8599,2.34738,8.5028,3.0563])
agent.bestcontrol
agent.bestrewards
