#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import keras 
from keras.models import Sequential
from keras.layers import Dense, Activation

def build_model_architecture():
    model = Sequential()
    model.add(Dense(200, input_shape=(376,)))
    model.add(Dense(17))
    model.compile(optimizer='rmsprop',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        for episode in xrange(10):
            if episode == 0:
                model = build_model_architecture()
            tf_util.initialize()

            import gym
            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            observations = []
            actions = []
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    if episode == 0:
                        action = policy_fn(obs[None,:])
                    else:
                        action = model.predict(obs[None,:])
                    observations.append(obs)
                    actions.append(action.reshape(17))
                    obs, r, done, _ = env.step(action.reshape(1,17))
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))

            expert_data = {'observations': np.array(observations),
                           'actions': np.array(actions)}

            print expert_data['observations'].shape
            print expert_data['actions'].shape
            
            model.fit(expert_data['observations'], expert_data['actions'])
            
            print expert_data

if __name__ == '__main__':
    main()
