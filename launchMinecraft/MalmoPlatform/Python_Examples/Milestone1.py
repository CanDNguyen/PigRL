from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Tutorial sample #2: Run simple mission using raw XML

from builtins import range
import MalmoPython
import os
import sys
import time
import json
import random
import logging
import numpy as np
import math
from pickle import NONE
from collections import defaultdict

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)
    

gamma = 1.0
lr = 1.0
min_lr = 0.002
epsilon = 0.02
max_vertical_degree = 20
max_horizontal_degree = 180
max_power = 10
eta = 0.0
class ArcherEnv(object):
    
    def __init__(self,my_mission):
        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.mission = my_mission
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.first_action = True
        self.action_selection = ["left","right","up","down", "power"]
        self.action = ""
        self.states = {} #to do
        self.recent_arrow_info = None
        self.target_info = None
        self.agent_info = None
        self.prev_s = None
        self.prev_a = None
        self.current_state = None
        
        self.trajectory = {} # state-action-reward sequence
        
        self.Qtable = None

        
        
        self.trans = defaultdict(lambda: defaultdict(float)) # transition probability distribution
    
    def updateQTable(self,reward,obs):
        
        dis = 5
        r = 0

        #print(obs[u'entities'])
        #print("last elememt at entities",obs[u'entities'][-1]['name'])
        
        for info in obs[u'entities']:
            if info[u'name'] == u'Pig':
                self.target_info = info
                break;
        if obs[u'entities'][-1]['name'] == u'Arrow':
            self.recent_arrow_info = obs[u'entities'][-1]
            print()
            print("{}:({}, {})".format(self.recent_arrow_info[u'name'],self.recent_arrow_info[u'x'],self.recent_arrow_info[u'z']))
            print()
   
        if None not in (self.recent_arrow_info,self.target_info):
            arrow_x = int(self.recent_arrow_info[u'x'])
            arrow_y = int(self.recent_arrow_info[u'z'])
            target_x = int(self.target_info[u'x'])
            target_y = int(self.target_info[u'z'])
            dis = math.sqrt( ( pow((target_x - arrow_x),2) + pow((target_y - arrow_y),2) ) )
        
        
        if reward == 0:
            r = dis * -1
        else:
            r = reward
            
        print("\nupdating:",self.prev_s)
        print("\nreward:",reward)
        print(self.Qtable[self.current_state].values())
        self.Qtable[self.prev_s][self.action] = self.Qtable[self.prev_s][self.action] + eta * \
                                                        (r + gamma * max(self.Qtable[self.current_state].values()) - self.Qtable[self.prev_s][self.action])
        self.__debuginfo__()
        return r
            
        #print("Qtable:",self.Qtable)
    def __convert_degree__(self,L):
        
        to_return = {}
       
        for k,v in L.items():
            if k == "power":
                to_return[k] = v / 10
            else:
                to_return[k] = v * 0.005

        return to_return
    
    def __debuginfo__(self):
        
        #for info in obs[u'entities']:
        #    if info[u'name'] == u'MalmoTutorialBot':
        #        print("Agent pos: ({}, {})\n".format(info[u'x'], info[u'z']))
        print("************************************************")
        if self.recent_arrow_info != None:
            print("last {}:({}, {})\n".format(self.recent_arrow_info[u'name'],self.recent_arrow_info[u'x'],self.recent_arrow_info[u'z']))
        print("\nprevious_state: ",self.prev_s)
        print("\ncurrent_state: ",self.current_state)
        print("\nTransition probability:\n")
        for key in self.trans.keys():
            print("state: ",key)
            for k,v in self.trans[key].items():
                print("{} -> {}%".format(k,v))
            print("-------------------------------------------")
            
        print("\nQtable:\n")
        for key in self.Qtable.keys():
            print("state: ",key)
            for k,v in self.Qtable[key].items():
                print("{} -> {}".format(k,v))
            print("-------------------------------------------")
        
        
        print("************************************************")
    def __adjust_degree__(self,prev_degree):
        
        print(self.recent_arrow_info)
        target_x = float(self.target_info[u'x'])
        target_y = float(self.target_info[u'z'])
        agent_x = float(self.agent_info[u'x'])
        agent_y = float(self.agent_info[u'z'])
        
        recent_arrow_pos = (float(self.recent_arrow_info[u'x']), float(self.recent_arrow_info[u'z']))
        adj_block_right = (float(self.recent_arrow_info[u'x']), float(self.recent_arrow_info[u'z']) - 2.0)
        adj_block_left = (float(self.recent_arrow_info[u'x']), float(self.recent_arrow_info[u'z']) + 2.0)
        adj_block_down = (float(self.recent_arrow_info[u'x']) + 1.0, float(self.recent_arrow_info[u'z']) )
        adj_block_up = (float(self.recent_arrow_info[u'x']) - 1.0, float(self.recent_arrow_info[u'z']) )
        
        dis_r = math.sqrt( ( pow((target_x - adj_block_right[0]),2) + pow((target_y - adj_block_right[1]),2) ) )
        dis_l = math.sqrt( ( pow((target_x - adj_block_left[0]),2) + pow((target_y - adj_block_left[1]),2) ) )
        dis_up = math.sqrt( ( pow((target_x - adj_block_up[0]),2) + pow((target_y - adj_block_up[1]),2) ) )
        dis_down = math.sqrt( ( pow((target_x - adj_block_down[0]),2) + pow((target_y - adj_block_down[1]),2) ) )
        
        blocks = {"right":dis_r, "left":dis_l, "up": dis_up, "down": dis_down}
        close_block = min(blocks, key=blocks.get)
        
        dis_recent_arrow = [recent_arrow_pos[0] - agent_x, recent_arrow_pos[1] - agent_y]
        norm_recent = math.sqrt(pow(dis_recent_arrow[0], 2) + pow(dis_recent_arrow[1], 2))
        direction_recent = [dis_recent_arrow[0] / norm_recent, dis_recent_arrow[1] / norm_recent]
        arrow_recent_vector = [direction_recent[0] * math.sqrt(2), direction_recent[1] * math.sqrt(2)]
        recent_vector_mag = np.linalg.norm(arrow_recent_vector)
        #print("recent: distance to agent",norm_recent)
        #print("recent",arrow_recent_vector)
        self.action = close_block
        #print("close_block",close_block)
        if close_block == "left":
            dis_left = [adj_block_left[0] - agent_x, adj_block_left[1] - agent_y]
            norm_left = math.sqrt(pow(dis_left[0], 2) + pow(dis_left[1], 2))
            direction_left = [dis_left[0] / norm_left, dis_left[1] / norm_left]
            arrow_left_vector = [direction_left[0] * math.sqrt(2), direction_left[1] * math.sqrt(2)]
            dot_left = np.dot(arrow_left_vector,arrow_recent_vector)
            left_vector_mag = np.linalg.norm(arrow_left_vector)
            degree_to_left = math.acos(dot_left / (left_vector_mag * recent_vector_mag) )
#             print("left: distance to agent:",norm_left)
#             print("left_vector",arrow_left_vector)
#             print("dot left", dot_left)
            print("degree to left:",degree_to_left)
            prev_degree[1] = degree_to_left * 100
            
        elif close_block == "right":
            dis_right = [adj_block_right[0] - agent_x, adj_block_right[1] - agent_y]
            norm_right = math.sqrt(pow(dis_right[0], 2) + pow(dis_right[1], 2))
            direction_right = [dis_right[0] / norm_right, dis_right[1] / norm_right]
            arrow_right_vector = [direction_right[0] * math.sqrt(2), direction_right[1] * math.sqrt(2)]
            dot_right = np.dot(arrow_right_vector,arrow_recent_vector)
            right_vector_mag = np.linalg.norm(arrow_right_vector)
            degree_to_right = math.acos(dot_right / (right_vector_mag * recent_vector_mag) )
            #print("right: distance to agent:",norm_right)
            #print("right_vector",arrow_right_vector)
            #print("dot right",dot_right)
            print("degree to right:",degree_to_right)
            prev_degree[1] = degree_to_right * 100
            
        elif close_block == "up":
            #prev_degree[0] = random.randint(0,10)
            prev_degree[2] = prev_degree[2] + 1
        else:
            #prev_degree[0] = random.randint(0,10)
            prev_degree[2] = prev_degree[2] - 1
        self.current_state = "{}:{}:{}".format(prev_degree[0],prev_degree[1],prev_degree[2])
        
        #dis_right = math.sqrt( ( pow((target_x - adj_block_right[0]),2) + pow((target_y - adj_block_right[1]),2) ) )
        #dis_left = math.sqrt( ( pow((target_x - adj_block_left[0]),2) + pow((target_y - adj_block_left[1]),2) ) )
        #recent_arrow_to_agent = math.sqrt(pow( (float(self.recent_arrow_info[u'x']) - agent_x),2) + pow((float(self.recent_arrow_info[u'z']) - agent_y),2))
        print("after adjust degree:",prev_degree)
        return prev_degree
                                
    def make_action(self, world_state):
        
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text) # most recent observation
        self.logger.debug(obs)
        #print(obs)
        degree = []
        res = []
        if self.agent_info == None:
            for info in obs[u'entities']:
                if info[u'name'] == u'MalmoTutorialBot':
                    self.agent_info = info
                    break
        
        # random version
        if obs[u'entities'][-1]['name'] == u'Arrow':
            self.recent_arrow_info = obs[u'entities'][-1]
            
        if self.first_action or (random.uniform(0,1) < epsilon):
            self.action = self.action_selection[random.randint(0,(len(self.action_selection))-2)]
            self.current_state = ""
            s = np.zeros(3)
            
           
            if self.action == "left" or self.action == "right":
                s[1] = random.randint(0,90)
            else:
                s[0] = random.randint(0,20)
                
            s[2] = random.randint(2,10)
            
            print()
            print("From random: current_state",self.current_state)
            print()
            self.current_state = "{}:{}:{}".format(s[0],s[1],s[2])
            self.Qtable[self.current_state] = {"left": 0,"right": 0, "up": 0, "down": 0}
            res = self.__convert_degree__({"vertical_degree":s[0],
                                          "horizontal_degree":s[1],
                                          "power":s[2]})
            self.first_action = False
      
        else:
            prev_degree = [float(i) for i in self.current_state.split(":")]
            logits = list(self.Qtable[self.prev_s].values())
            print(logits)
            logits_exp = np.exp(logits)
            probs = logits_exp / np.sum(logits_exp)
            i = 0
            for k in self.trans[self.prev_s]:
                self.trans[self.prev_s][k] = probs[i]
                i += 1
            
            print(self.trans[self.prev_s])
            self.action = np.random.choice([key for key in self.trans[self.prev_s].keys() if self.trans[self.prev_s][key] == max(self.trans[self.prev_s].values())])
            degree = self.__adjust_degree__(prev_degree)
            self.current_state = "{}:{}:{}".format(degree[0],degree[1],degree[2])
            self.Qtable[self.current_state] = {"left": 0,"right": 0, "up": 0, "down": 0}
            res = self.__convert_degree__({"vertical_degree": degree[0],
                                          "horizontal_degree":degree[1],
                                          "power":degree[2]})
        print("Action: ",self.action)
        print("Commands: ",res)
        print()
        return res
    
    def turn_up_down(self,agent_host,commands):
        
        if commands != 0:
            if self.action == "up":
                agent_host.sendCommand("pitch -1")
                time.sleep(abs(commands))
                agent_host.sendCommand("pitch 0")
            elif self.action == "down":
                agent_host.sendCommand("pitch 1")
                time.sleep(abs(commands))
                agent_host.sendCommand("pitch 0")
            
    def turn_left_right(self,agent_host,commands):
        
        if commands != 0:
            if self.action == "right":
                agent_host.sendCommand("turn 1")
                time.sleep(abs(commands))
                agent_host.sendCommand("turn 0")
            elif self.action == "left":
                agent_host.sendCommand("turn -1")
                time.sleep(abs(commands))
                agent_host.sendCommand("turn 0")
            
    def execute_action(self,agent_host,commands):

        self.turn_left_right(agent_host,commands["horizontal_degree"])
        self.turn_up_down(agent_host, commands["vertical_degree"])
        
        time.sleep(0.2)
        agent_host.sendCommand("use 1")
        time.sleep(commands["power"])
        agent_host.sendCommand("use 0")
        time.sleep(1)
        
    def run(self, agent_host):
        
        total_discounted_reward = 0
        total_step = 30
        step_idx = 0
        self.Qtable = defaultdict(lambda: defaultdict(float))
        world_state = agent_host.getWorldState()
        while world_state.is_mission_running:
            
            for i in range(total_step):
                if self.first_action:
                        current_r = 0
                        self.prev_s = "0:0:0"
                        self.trans[self.prev_s] = {"left": 0,"right": 0, "up": 0, "down": 0}
                        self.Qtable[self.prev_s] = {"left": 0,"right": 0, "up": 0, "down": 0}
                        time.sleep(0.1)
                        world_state = agent_host.getWorldState()
                        for error in world_state.errors:
                            self.logger.error("Error: %s" % error.text)
                            
                        if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                            self.execute_action(agent_host, self.make_action(world_state))
                            for reward in world_state.rewards:
                                #print(reward)
                                #print("reward.getValue()",reward.getValue())
                                current_r += reward.getValue()
                            obs_text = world_state.observations[-1].text
                            obs = json.loads(obs_text)
                            total_discounted_reward += self.updateQTable(current_r, obs)
                        
                else:
                    current_r = 0
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                            
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        self.execute_action(agent_host, self.make_action(world_state))
                        for reward in world_state.rewards:
                            #print(reward)
                            print("\nreward.getValue()",reward.getValue())
                            current_r += reward.getValue()
                        obs_text = world_state.observations[-1].text
                        obs = json.loads(obs_text)
                        total_discounted_reward += self.updateQTable(current_r, obs)
                        self.prev_s = self.current_state
                        self.trans[self.prev_s] = {"left": 0,"right": 0, "up": 0, "down": 0}
                        
         
# More interesting generator string: "3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"
def drawrailline(x1, z1, x2, z2, y):
    ''' Draw a powered rail between the two points '''
    y1railsegment = '" y1="' + str(y) + '" z1="'
    y2railsegment = '" y2="' + str(y) + '" z2="'
    y1stonesegment = '" y1="' + str(y-1) + '" z1="'
    y2stonesegment = '" y2="' + str(y-1) + '" z2="'
    shape = "north_south"
    if z1 == z2:
        shape = "east_west"
    rail_line = '<DrawLine x1="' + str(x1) + y1railsegment + str(z1) + '" x2="' + str(x2) + y2railsegment + str(z2) + '" type="golden_rail" variant="' + shape + '"/>'
    redstone_line = '<DrawLine x1="' + str(x1) + y1stonesegment + str(z1) + '" x2="' + str(x2) + y2stonesegment + str(z2) + '" type="redstone_block" />'
    #return redstone_line + rail_line
    #return rail_line
    return redstone_line

def drawblock(x, y, z):
    ''' Draw a corner piece of rail '''
    shape = ""
    if z < 0:
        if x > 0:
            shape="south_west"
        else:
            shape="south_east"
    else:
        if x > 0:
            shape="north_west"
        else:
            shape="north_east"

    yrailsegment = '" y="' + str(y) + '" z="'
    ystonesegment = '" y="' + str(y-1) + '" z="'

    #return '<DrawBlock x="' + str(x) + ystonesegment + str(z) + '" type="redstone_block"/>' + \
    #       '<DrawBlock x="' + str(x) + yrailsegment + str(z) + '" type="rail" variant="' + shape + '"/>'
    #return '<DrawBlock x="' + str(x) + yrailsegment + str(z) + '" type="rail" variant="' + shape + '"/>'
    return '<DrawBlock x="' + str(x) + ystonesegment + str(z) + '" type="redstone_block"/>'
def drawloop(radius, y):
    ''' Create a loop of powered rail '''
    return drawrailline(-radius, 1-radius, -radius, radius-1, y) + \
           drawrailline(radius, 1-radius, radius, radius-1, y) + \
           drawrailline(1-radius, radius, radius-1, radius, y) + \
           drawrailline(1-radius, -radius, radius-1, -radius, y) + \
           drawblock(radius, y, radius) + drawblock(-radius, y, radius) + drawblock(-radius, y, -radius) + drawblock(radius, y, -radius)
           
           

missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Hello world!</Summary>
              </About>
              
              <ServerSection>
                <ServerInitialConditions>
                  <Time>
                    <StartTime>12000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                  </Time>
                  <Weather>clear</Weather>
                </ServerInitialConditions>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,56*35:9,36;,biome_1"/>
                  <DrawingDecorator>
                      '''+drawloop(5, 57)+'''
                      <DrawEntity x="0" y="57" z="2" type="Pig"/>
                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="300000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>MalmoTutorialBot</Name>
                <AgentStart>
                    <Placement x="15" y="57" z="0" yaw="90"/>
                    <Inventory>
                        <InventoryItem slot="0" type="bow"/>
                        <InventoryItem slot="1" type="arrow" quantity="60"/>
                    </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromNearbyEntities>
                     <Range name="entities" xrange="60" yrange="40" zrange="60"/>
                  </ObservationFromNearbyEntities>
                  <RewardForDamagingEntity>
                      <Mob type ="Pig" reward="100"/>
                  </RewardForDamagingEntity>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

# Create default Malmo objects:

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()

agent = ArcherEnv(my_mission)
num_of_repeats = 10
# Attempt to start a mission:
max_retries = 3
for i in range(num_of_repeats):
    print()
    print('Repeat %d of %d' % ( i+1, num_of_repeats ))
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)
    
    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    eta = max(min_lr,lr * (0.85 ** (i//100)))
    agent.run(agent_host)
    time.sleep(1)

print("Mission running ", end=' ')


# Loop until mission ends:
# agent_host.sendCommand("turn -1")
# time.sleep(0.1)
# agent_host.sendCommand("turn 0")
# while world_state.is_mission_running:
#     print(".", end="")
#     time.sleep(0.1)
#     agent_host.sendCommand("setYaw 30")
#
#     world_state = agent_host.getWorldState()
#     for error in world_state.errors:
#         print("Error:",error.text)
#     if len(world_state.observations) > 0:
#         obs_text = world_state.observations[-1].text
#         obs = json.loads(obs_text) # most recent observation


print()
print("Mission ended")
# Mission has ended.

