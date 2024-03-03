import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_json_file
from trackmania_env import TrackmaniaEnv
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_actions = 3
env = TrackmaniaEnv(num_actions=num_actions)

max_available_time = 15

n_actions = env.action_space.n
# state, info = env.reset()
state = [0, 0, 0, 0, 0, 0]
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)

policy_net_state_dict = torch.load(r'D:\study\bachelor\github\trackmania-ai\models\rl_models\included_validation\best_policy_dqn_model_epoch_0.pt')
target_net_state_dict = torch.load(r'D:\study\bachelor\github\trackmania-ai\models\rl_models\included_validation\best_target_dqn_model_epoch_0.pt')

policy_net.load_state_dict(policy_net_state_dict)
target_net.load_state_dict(target_net_state_dict)

# car position right: 
# car position left: 688
currect_observation = [672, 688, 10]
previous_observation = [670, 688, 10]

state = currect_observation + previous_observation
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

print(state)
print(target_net(state))

src_path = r'D:\study\bachelor\temp-files\stored txt angelscript files\output.txt'

previous_observation = [0, 0, 0]

while True:
    while True:
        try:
            json_data = load_json_file(src_path)
            break
        except:
            print("PROBLEM WITH JSON FILE")
            time.sleep(0.05)
            continue
    
    if 'carPositionRight' not in json_data:
        continue
    
    currect_observation = [json_data['carPositionRight'], json_data['carPositionLeft'], json_data['carPositionHeight']]

    # if currect_observation[0] - previous_observation[0] < 1 and currect_observation[1] - previous_observation[1]:
    #     continue

    state = currect_observation + previous_observation
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    policy_actions = policy_net(state_tensor)
    action = policy_actions.max(1).indices.view(1, 1).item()
    print(policy_actions.tolist(), f"Action: {action}")

    previous_observation = currect_observation

    # time.sleep(3)