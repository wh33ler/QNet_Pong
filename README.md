# pong_neural_net_live

## Deep Q Network Example
This code is originaly by [Siraj](https://github.com/llSourcell/pong_neural_network_live) and  [malreddysid](https://github.com/malreddysid)
It is a simple implemention of a Deep Q Network which learns to play PONG against a perfectly playing computer

The basis of this code was good but had some issues that didn't work for me. 
So I fixed them as follows: 

## Changes
I did some changes to the network and to the pong game
* fixed for python3 and tensorflow v1
* added some text overlay in the game to reduce prints in console
* added a positive reward when agent hits the ball
* better weight and bias initialisation of the conv net
* added a pooling layer
* shrinked the size of some layers for fast computing
* reduced batch size to 32 for performance reasons
* Reduced Explore and Observation range
* Fixed saving and reloading from checkpoints 
* uploaded checkpoints and a model_mode to see the result 

## Results
After around 700k Time Steps the agent plays a near perfect game and even found some bugs of the PONG implementation to exploit. First I thought it was just not good enough until I saw that it was just a way of receiving more points by moving the paddle a certain way. Really hard to expoit by hand.... but see by yourself. It is kind of eye opening what a learning machine is capebale of.


## Original README Below

## Overview
This is the code for the Live [Youtube](https://www.youtube.com/watch?v=Hqf__FlRlzg) session by @Sirajology. In this live session I build
the game of [Pong](http://www.ponggame.org) from scratch. Then I build a [Deep Q Network](https://www.quora.com/Artificial-Intelligence-What-is-an-intuitive-explanation-of-how-deep-Q-networks-DQN-work) that gets better and better over time through trial and error. The DQN is a convolutional neural network that reads in pixel data from the game and the game score. Using just those 2 parameters, it learns what moves it needs to make to become better.

## Installation


* tensorflow (https://www.tensorflow.org)
* cv2 (http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/)
* numpy
* random
* collections
* pygame

use [pip](https://pypi.python.org/pypi/pip) to install the dependencies. Tensorflow and cv2 are more manual. Links provided above ^

## Usage 

Run it like this in terminal. The longer you let it run, the better it will get.

```
python RL.py
```

## Credits

This code was by [malreddysid](https://github.com/malreddysid) i've merely wrapped, updated, and documented it. 

