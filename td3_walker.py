import numpy as np
import sys
import gym
import torch
import random
from td3 import TD3
from buffer import ExperienceReplay
import matplotlib.pyplot as plt
import PIL
import IPython as ip

def visualize_episode(env, model, name):
    """
    generates arrays of states, actions, and rewards for one complete episode.

    env:        the openai gym environment
    model:      the model used to generate the actions
    returns:    list of image frames
    """

    state = env.reset()[0]
    image_frames = []
    max_steps_per_episode = env.spec.max_episode_steps
    
    for step in range(max_steps_per_episode):
        state = torch.FloatTensor(state.reshape(1, -1)).to(torch.device("cpu"))
        action = model.actor(state).cpu().data.numpy().flatten()
        
        state, reward, done, _, _ = env.step(action)
        
        image_array = env.render()
        image_frame = PIL.Image.fromarray(image_array)
        image_frames.append(image_frame)
        
        if done:
            break

    image_frames[0].save(name + '.gif', 
                        save_all = True, 
                        duration = 20,
                        loop = 0,
                        append_images = image_frames[1:])

    ip.display.Image(open(name + '.gif', 'rb').read())

def main():
    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')

    # set seed for reproducable results
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    buffer_size = 1000000
    batch_size = 100
    noise = 0.1

    # Uncomment to use GPU, but errors exist if GPU is not supported anymore.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    device = torch.device("cpu")

    policy = TD3(state_dim, action_dim, max_action, env, device)
    
    try:
        print("loading model")
        policy.load()
    except Exception:
        print('training new model')

    print('simulating pre-trained model')
    visualize_episode(env, policy, 'untrained_walker')

    buffer = ExperienceReplay(buffer_size, batch_size, device)

    save_score = 400
    episodes = 5
    timesteps = 2000

    best_reward = -1 * sys.maxsize
    scores_over_episodes = []

    for episode in range(episodes):
        avg_reward = 0
        state = env.reset()[0]
        for i in range(timesteps):
            # Same as the TD3, select an action and add noise:
            action = policy.select_action(state) + np.random.normal(0, max_action * noise, size=action_dim)
            action = action.clip(env.action_space.low, env.action_space.high)
            # Make an action. 
            next_state, reward, done, _, _ = env.step(action)
            buffer.store_transition(state, action, reward, next_state, done)
            state = next_state
            avg_reward += reward
            env.render()
            if(len(buffer) > batch_size):
                policy.train(buffer, i)
            if(done or i > timesteps):
                scores_over_episodes.append(avg_reward)
                print('episode', episode, ':', avg_reward)
                break
        
        if(np.mean(scores_over_episodes[-50:]) > save_score):
            print('saving agent- : past 50 sma >', save_score)
            best_reward = np.mean(scores_over_episodes[-50:])
            save_score = best_reward
            policy.save()
            break # Saved agent. Break out of episodes and end, 400 is pretty good. 

        if(episode >= 0 and avg_reward > best_reward):
            print('saving agent : new best reward of', avg_reward)
            best_reward = avg_reward
            policy.save() # Save current policy + optimizer
    
    print('simulating trained model')
    policy.load()
    visualize_episode(env, policy, 'trained_walker')
    
    plt.figure()
    plt.plot(np.arange(1, len(scores_over_episodes) + 1), scores_over_episodes)
    plt.ylabel('score')
    plt.xlabel('episode #')
    plt.show()


main()