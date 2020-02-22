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
epsilon = 0.4
max_vertical_degree = 20
max_horizontal_degree = 180
max_power = 10
eta = 0.0
target_info = None
num_of_hits = 0

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
        self.first_state = None
        self.action_selection = ["yaw","pitch", "power"]
        self.action = ""
        self.degree = None
        self.states = {} #to do
        self.recent_arrow_info = None
        self.target_info = None
        self.agent_info = None
        self.prev_s = None
        self.current_state = None
        self.trajectory = {} # state-action-reward sequence
        self.shots = 0
        self.Qtable = defaultdict(lambda: defaultdict(float))
        self.first_round = True

        
        
        self.trans = defaultdict(lambda: defaultdict(float)) # transition probability distribution 
    def reset(self):
        self.prev_s = self.first_state
        self.shots = 0
        
    def updateQTable(self,reward):

        dis = 0
        r = 0
        if None not in (self.recent_arrow_info,target_info): 
            arrow_x = int(self.recent_arrow_info[u'x'])
            arrow_y = int(self.recent_arrow_info[u'z'])
            target_x = int(target_info[u'x'])
            target_y = int(target_info[u'z'])
            dis = math.sqrt( ( pow((target_x - arrow_x),2) + pow((target_y - arrow_y),2) ) )
            dis = round(dis,2)
        
        if reward > 0:
            r = reward
        else:
            r = -1 * dis
        
     
        #print("\nreward:",reward)
        #print(self.Qtable[self.current_state].values())
        if not self.first_action:
            #print('self.degree in updateQtable',self.degree)
            self.Qtable[self.prev_s][self.degree] =  (1-eta) * self.Qtable[self.prev_s][self.degree] + eta * \
                                                        (r + gamma * max(self.Qtable[self.current_state].values()) - self.Qtable[self.prev_s][self.degree])
        else:
            #print('self.degree in updateQtable',self.degree)
            self.first_action = False
            self.first_state = self.current_state
            self.Qtable[self.current_state][self.degree] = 0 + eta * (r + gamma * max(self.Qtable[self.current_state].values())) - 0

        #self.__debuginfo__()

        return r
            
        #print("Qtable:",self.Qtable)
    
    def __debuginfo__(self):
        
        #for info in obs[u'entities']:
        #    if info[u'name'] == u'MalmoTutorialBot':
        #        print("Agent pos: ({}, {})\n".format(info[u'x'], info[u'z']))
        print("************************************************")
#         if self.recent_arrow_info != None:
#             print("last {}:({}, {})\n".format(self.recent_arrow_info[u'name'],self.recent_arrow_info[u'x'],self.recent_arrow_info[u'z']))
#         print("\nprevious_state: ",self.prev_s)
#         print("\ncurrent_state: ",self.current_state)
#         print("\nTransition probability:\n")
#         for key in self.trans.keys():
#             print("state: ",key)
#             for k,v in self.trans[key].items():
#                 print("{} -> {}%".format(k,v))
#             print("-------------------------------------------")
            
        print("\nQtable:\n")
        for key in self.Qtable.keys():
            print("state: ",key)
            for k,v in self.Qtable[key].items():
                print("{} -> {}".format(k,v))
            print("-------------------------------------------")
        
        
        print("************************************************")
    def __adjust_action__(self):
        
        print(self.recent_arrow_info)
        target_x = float(self.target_info[u'x'])
        target_y = float(self.target_info[u'z'])
        agent_x = float(self.agent_info[u'x'])
        agent_y = float(self.agent_info[u'z'])
        
        recent_arrow_pos = (float(self.recent_arrow_info[u'x']), float(self.recent_arrow_info[u'z']))
        adj_block_right = (float(self.recent_arrow_info[u'x']), float(self.recent_arrow_info[u'z']) - 1.0)
        adj_block_left = (float(self.recent_arrow_info[u'x']), float(self.recent_arrow_info[u'z']) + 1.0)
        adj_block_down = (float(self.recent_arrow_info[u'x']) + 1.0, float(self.recent_arrow_info[u'z']) )
        adj_block_up = (float(self.recent_arrow_info[u'x']) - 1.0, float(self.recent_arrow_info[u'z']) )
        
        dis_r = math.sqrt( ( pow((target_x - adj_block_right[0]),2) + pow((target_y - adj_block_right[1]),2) ) ) + float(self.recent_arrow_info[u'y'])
        dis_l = math.sqrt( ( pow((target_x - adj_block_left[0]),2) + pow((target_y - adj_block_left[1]),2) ) ) + float(self.recent_arrow_info[u'y'])
        dis_up = math.sqrt( ( pow((target_x - adj_block_up[0]),2) + pow((target_y - adj_block_up[1]),2) ) ) + float(self.recent_arrow_info[u'y'])
        dis_down = math.sqrt( ( pow((target_x - adj_block_down[0]),2) + pow((target_y - adj_block_down[1]),2) ) ) + float(self.recent_arrow_info[u'y'])
        arrow_to_target = math.sqrt( ( pow((target_x - recent_arrow_pos[0]),2) + pow((target_y - recent_arrow_pos[1]),2) ) ) + float(self.recent_arrow_info[u'y'])
        
        blocks = {"right":dis_r, "left":dis_l, "up": dis_up, "down": dis_down}
        close_block = min(blocks, key=blocks.get)
        return [close_block,arrow_to_target]
        
                                
    def make_action(self, world_state):
    
        degree = []
        res = []
        degree_factor = 0
                        
        if self.first_action or (random.uniform(0,1) < epsilon):
            self.current_state = ""
            s = np.zeros(3)
            s[1] = random.randint(82,98)
            s[0] = random.randint(-3,0) 
            s[2] = 10   
            self.action = self.action_selection[random.randint(0,len(self.action_selection) - 2)]
            
            self.current_state = "Arrow_{}".format(self.shots)
            #self.current_state = "{}:{}:{}".format(s[0],s[1],s[2])   
            print()
            print("From random: current_state",self.current_state)
            print()
            #if(self.current_state not in self.Qtable.keys()):
            self.degree = "{}:{}:{}:{}".format(s[0],s[1],s[2],self.action)
            if self.degree not in self.Qtable[self.current_state].keys():
                #print('self.degree in make_action',self.degree)
                self.Qtable[self.current_state][self.degree] = 0
            
            return s
        else:
            #print("self.previous_state",self.prev_s)
            best_action = np.random.choice([key for key in self.Qtable[self.prev_s].keys() if self.Qtable[self.prev_s][key] == max(self.Qtable[self.prev_s].values())])   
            l = best_action.split(":")
            prev_degree = [float(l[i]) for i in range(0,len(l)-1)]
            self.action = l[-1]
            
            if self.target_info != None and self.recent_arrow_info != None:
                close_block,dis = self.__adjust_action__()
                print("prev_degree[1] + 5 + degree_factor = ",prev_degree[1] + 5 + degree_factor)
                print("prev_degree[1] - 5 + degree_factor = ",prev_degree[1] + 5 + degree_factor)
                print("close block:",close_block)
                print("degree_factor",degree_factor)
    
              
                if close_block == "right" and (prev_degree[1] + 2) <= 98:
                    prev_degree[1] += 2
                    print("+3")
                elif close_block == "left" and (prev_degree[1] - 2 ) >= 82:
                    prev_degree[1] -= 2
                    print("-3")
                elif close_block == "up" and (prev_degree[0] - 1) >= -3:
                    prev_degree[0] -= 1
                    print("-1")
                elif close_block == "down" and (prev_degree[0] + 1)  <= 0:
                    prev_degree[0] += 1
                    print("+1")
            
            
            self.current_state = "Arrow_{}".format(self.shots)
            self.degree = "{}:{}:{}:{}".format(prev_degree[0],prev_degree[1],prev_degree[2],self.action)
            #if(self.current_state not in self.Qtable.keys()):
            if self.degree not in self.Qtable[self.current_state].keys():
                #print('self.degree in make_action',self.degree)
                self.Qtable[self.current_state][self.degree] = 0
            return prev_degree
            
    def get_recent_obs(self):
        
        while True:
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                obs_text = world_state.observations[-1].text
                obs = json.loads(obs_text) # most recent observation
                self.logger.debug(obs)
                
                for info in obs[u'entities']:
                    if info[u'name'] == u'Pig':
                        self.target_info = info
                        break
                    else:
                        self.target_info = None
                        
                for info in obs[u'entities']:
                    if info[u'name'] == u'MalmoTutorialBot':
                        self.agent_info = info
                        break
                    
                if obs[u'entities'][-1]['name'] == u'Arrow':
                    self.recent_arrow_info = obs[u'entities'][-1]
                    print()
                    print("Recent {}:({}, {})".format(self.recent_arrow_info[u'name'],self.recent_arrow_info[u'x'],self.recent_arrow_info[u'z']))
                    print()
                break
           
        
        return world_state
    
    def execute_action(self,agent_host,commands):

        if self.action == "yaw":
            print("taking aciton: yaw:",commands[1])
            agent_host.sendCommand("setYaw {}".format(commands[1]))
        else:
            print("taking aciton: pitch:",commands[0])
            agent_host.sendCommand("setPitch {}".format(commands[0]))
  
        time.sleep(0.1)
        agent_host.sendCommand("use 1")
        time.sleep(commands[2] / 10)
        agent_host.sendCommand("use 0")
        time.sleep(0.5)
        
    def run(self, agent_host):
        total_discounted_reward = 0
        total_step = 30
        step_idx = 0
        world_state = agent_host.getWorldState()
        global num_of_hits
        while world_state.is_mission_running:  
            for i in range(total_step):
                current_r = 0
                if self.first_action:
                    while True:
                        time.sleep(0.1)
                        world_state = agent_host.getWorldState()
                        for error in world_state.errors:
                            self.logger.error("Error: %s" % error.text)
                            
                        if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                            self.shots += 1
                            world_state = self.get_recent_obs()
                            if self.target_info == None:
                                return (self.shots,total_discounted_reward)
                            time.sleep(0.1)
                            self.execute_action(agent_host, self.make_action(world_state))
                            time.sleep(0.1)
                            world_state = self.get_recent_obs()
                            self.prev_s = self.current_state
                            #world_state = agent_host.getWorldState()
                            
                            if world_state.number_of_rewards_since_last_state > 0:
                                current_r = sum(reward.getValue() for reward in world_state.rewards)
                                print("current reward:",current_r)
                                
                            total_discounted_reward += (gamma ** i) * self.updateQTable(current_r)
                            break
                                
                else:              
                    while True: 
                        time.sleep(0.1)
                        world_state = agent_host.getWorldState()
                        for error in world_state.errors:
                            self.logger.error("Error: %s" % error.text)
                        
                        if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                            self.shots += 1
                            world_state = self.get_recent_obs()
                            if self.target_info == None:
                                return (self.shots,total_discounted_reward)
                            time.sleep(0.1)
                            self.execute_action(agent_host, self.make_action(world_state))             
                            world_state = self.get_recent_obs()
                            if world_state.number_of_rewards_since_last_state > 0:
                                current_r = sum(reward.getValue() for reward in world_state.rewards)
                            total_discounted_reward += (gamma ** i) * self.updateQTable(current_r)
                            self.prev_s = self.current_state
                            break
        
                if current_r > 0:
                    num_of_hits += 1
                    print("current reward:",current_r)
                    print("num_of_hits",num_of_hits)
                
                
                        
            return (30,total_discounted_reward)
           
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
    print(drawrailline(-radius, 1-radius, -radius, radius-1, y))
    return drawrailline(-radius, 1-radius, -radius, radius-1, y) + \
           drawrailline(1-radius, radius, radius-1, radius, y) + \
           drawrailline(1-radius, -radius, radius-1, -radius, y) + \
           drawrailline(-radius, 1-radius, -radius, radius-1, y+1) + \
           drawrailline(1-radius, radius, radius-1, radius, y+1) + \
           drawrailline(1-radius, -radius, radius-1, -radius, y+1) + \
           drawblock(radius, y, radius) + drawblock(-radius, y, radius) + drawblock(-radius, y, -radius) + drawblock(radius, y, -radius)
           #drawrailline(radius, 1-radius, radius, radius-1, y) + \ at second
           
def random_pos():
    L = [(0,0),(15,15),(15,-15),(30,0)]
    i = random.randint(0,len(L)-1)
    x,z = L[0]
    print("target pos: ({}, {})".format(x,z))
    return '<DrawEntity x="'+str(x)+'" y="57" z="' + str(z) + '" type="Pig" />'
    
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
                      '''+drawloop(5, 58)+'''
                      '''+drawloop(5, 59)+'''
                      '''+random_pos()+'''
                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="300000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>MalmoTutorialBot</Name>
                <AgentStart>
                    <Placement x="15" y="57" z="0" yaw="75"/>
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
                      <Mob type ="Pig" reward="10"/>
                  </RewardForDamagingEntity>
                  <ChatCommands />
                  <MissionQuitCommands quitDescription="give_up"/>
                  <ObservationFromFullStats/>
                  <AbsoluteMovementCommands/>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

missionXML_2 ='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
                  <ChatCommands />
                  <MissionQuitCommands quitDescription="give_up"/>
                  <ObservationFromFullStats/>
                  <AbsoluteMovementCommands/>
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
my_mission_2 = MalmoPython.MissionSpec(missionXML_2, True)


agent = ArcherEnv(my_mission)
num_of_repeats = 50
cumulative_rewards = []
total_shots = 0
# Attempt to start a mission:
max_retries = 3
target_dead = False
for i in range(num_of_repeats):
    for retry in range(max_retries):
        try:
            if i == 0 or target_dead:
                agent_host.startMission( my_mission, my_mission_record )
            else:
                agent_host.startMission( my_mission_2, my_mission_record )
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
   
            
    
    print()
    print('Repeat %d of %d' % ( i+1, num_of_repeats ))

    #print("target_info",target_info)        
    eta = max(min_lr,lr * (0.85 ** (i//100)))
    if epsilon > 0:
        epsilon -= 0.01 * i
    num_of_shots,cumulative_reward = agent.run(agent_host)
    total_shots += num_of_shots
    print("total shots:",total_shots)
    print("num of hits",num_of_hits)
    if total_shots != 0:
        print('Accurancy: {}%'.format(round((num_of_hits/total_shots),3) * 100))
    print('Cumulative reward: %d' % cumulative_reward)
    cumulative_rewards += [ cumulative_reward ]
    agent.reset()
    
    while True:
        world_state = agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            obs_text = world_state.observations[-1].text
            obs = json.loads(obs_text)
            target_dead = True
            for info in obs[u'entities']:
                if info[u'name'] == u'Pig':
                    target_dead = False
                    break
            break
 
    time.sleep(0.1)
    agent_host.sendCommand("quit")
    time.sleep(1)

print("total episodes reward:",cumulative_rewards)

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
