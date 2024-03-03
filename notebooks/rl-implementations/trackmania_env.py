import random
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pynput.keyboard import Key, Controller
from directkeys import DELETE, UP, LEFT, RIGHT
from directkeys import PressKey, ReleaseKey

from utils import load_json_file

class TrackmaniaActionSpace:
    def __init__(self, num_actions):
        self.n = num_actions

    def sample(self):
        # 0 - left and forward
        # 1 - forward
        # 2 - right and forward
        return random.randint(0, self.n - 1)

    def __call__(self):
        return self.sample()
    

class TrackmaniaEnv:
    def __init__(self, num_actions):
        self.multiplier = 1
        self.controller = Controller()
        self.action_space = TrackmaniaActionSpace(num_actions)
        self.observation = [0, 0, 0] # x, y, z 
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.src_path = r'D:\study\bachelor\temp-files\stored txt angelscript files\output.txt'
        self.info = {
            "previous_speed": -1,
            }
        
        self.previous_observation = [0, 0, 0] # prev_x, prev_y, prev_z
        self.previous_data = None 

    def reset(self):

        self.controller.press(Key.delete)
        self.controller.release(Key.delete)

        self.controller.release(Key.up)
        self.controller.release(Key.left)
        self.controller.release(Key.right)

        self.controller.press(Key.up)

        self.observation = [0, 0, 0]
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.info = {
            "last_10_speeds": [],
            "previous_x": 0,
            "previous_y": 0,
            "previous_z": 0,
            }
        self.previous_data = None
        self.previous_observation = [0, 0, 0]

        print("Environment reset")

        observation = self.observation + self.previous_observation
        return observation, self.info
    

    @staticmethod
    def is_speed_increasing(current_speed, previous_speed):
        if current_speed > previous_speed:
            return True

        return False


    def is_bonk(self, current_speed):
        self.info['last_10_speeds'].append(current_speed)
        
        if len(self.info['last_10_speeds']) > 10:
            self.info['last_10_speeds'].pop(0)
        
        avg_speed = sum(self.info['last_10_speeds']) / len(self.info['last_10_speeds'])

        if avg_speed < 10 and len(self.info['last_10_speeds']) > 9:
            return True
        return False


    def step(self, action, reward_time):
        # print("NEXT STEP")
    
        self.complete_action(action) 

        time.sleep(0.07)
        
        while True:
            try:
                json_data = load_json_file(self.src_path)
                break
            except:
                print("PROBLEM WITH JSON FILE")
                continue

        if self.previous_data is not None:
            if not self.is_speed_increasing(current_speed=json_data['speed'],
                                            previous_speed=self.previous_data['speed']):
                self.reward -= 0.5

        self.observation[0] = json_data['carPositionRight']
        self.observation[1] = json_data['carPositionLeft']
        self.observation[2] = json_data['carPositionHeight']

        # print(self.observation)
        # print(self.previous_observation)
        observation = self.observation + self.previous_observation

        if json_data['is_crash']:
            self.reward -= 1
            self.truncated = True

            print(f"DONE crash, reward: {self.reward}, is_terminated: {json_data['is_terminated']}")

            return observation, self.reward, self.terminated, self.truncated, self.info

        is_bonk_value = self.is_bonk(json_data['speed'])
        if json_data['is_terminated'] == True or is_bonk_value:
            print(f"DONE, reward: {self.reward}, is_bonk: {is_bonk_value}, is_terminated: {json_data['is_terminated']}")

            self.truncated = is_bonk_value
            self.terminated = json_data['is_terminated']

            if not is_bonk_value:
                self.reward += reward_time * self.multiplier

            return observation, self.reward, self.terminated, self.truncated, self.info
        
        
        # if json_data['is_terminated'] == True:
        #     print(f"DONE: {reward_time}")
        #     self.terminated = True
        #     self.reward += reward_time
        #     return self.observation, self.reward, self.terminated, self.truncated, self.info
        
        # if self.is_bonk(json_data['speed']):
        #     print(f"Episode truncated due to bonk: {reward_time}")
        #     self.truncated = True
        #     return self.observation, self.reward, self.terminated, self.truncated, self.info

        if json_data == self.previous_data:
            print("MISDATAAAAAAAAAAAAAAAAAAAAa")
        
        # self.reward = json_data['speed']
        self.info['previous_speed'] = json_data['speed']

        self.previous_data = json_data
        self.previous_observation = self.observation

        return observation, self.reward, self.terminated, self.truncated, self.info
    
    # TODO update obser, reward, trunc etc
    def complete_action(self, action):
        if action == 0:
            self.controller.release(Key.right)
            self.controller.press(Key.left)
            # ReleaseKey(RIGHT)
            # PressKey(LEFT)
        elif action == 1:
            self.controller.release(Key.right)
            self.controller.release(Key.left)
            # ReleaseKey(RIGHT)
            # ReleaseKey(LEFT)
        elif action == 2:
            self.controller.press(Key.right)
            self.controller.release(Key.left)
            # ReleaseKey(LEFT)
            # PressKey(RIGHT)
