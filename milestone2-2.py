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
from fileinput import close

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)
    

gamma = 1.0
lr = 1.0
min_lr = 0.002
epsilon = 0.6
max_vertical_degree = 20
max_horizontal_degree = 180
max_power = 10
eta = 0.0
target_info = None
num_of_hits = 0

class ArcherEnv(object):
    
    def __init__(self,y1,y2,p1,p2):
        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.first_action = True
        self.first_state = None
        #self.action_selection = ["yaw","pitch", "power"]
        #self.action = ""
        self.states = dict()
        self.prev_degree = None
        self.recent_arrow_info = None
        self.target_info = None
        self.agent_info = None
        self.prev_s = None
        self.current_state = None
        self.shots = 0
        self.target_health = None
        self.Qtable = defaultdict(lambda: defaultdict(float))
        self.yaw_range = (y1,y2)
        self.pitch_range = (p1,p2)
        
    def get_target_info(self,t):
        self.target_info = t
        self.target_health = int(self.target_info[u'life'])
        
    def reset(self):
        self.current_state = None
        self.shots = 0
        self.prev_s = None
        self.prev_degree = None
        
    def updateQTable(self,reward):

        print("\nraw reward:",reward)
        #print(self.Qtable[self.current_state].values())
        
        print("updating",self.prev_s)
        print("degree",self.prev_degree)
  
        
        self.Qtable[self.prev_s][self.prev_degree] = (1-eta) * self.Qtable[self.prev_s][self.prev_degree] + eta * \
                                                    (reward + gamma * max(self.Qtable[self.current_state].values()) - self.Qtable[self.prev_s][self.prev_degree])
                                                    
        

        #self.__debuginfo__()
        for k,v in self.Qtable[self.prev_s].items():
            print("{} -> {}".format(k,v))
        return reward
            
        #print("Qtable:",self.Qtable)
    
    def __debuginfo__(self):
        
        print("************************************************")

        print("\nQtable:\n")
        for key in self.Qtable.keys():
            print("state: ",key)    
            for k,v in self.Qtable[key].items():
                print("{} -> {}".format(k,v))
            print("-------------------------------------------")
        
        
        print("************************************************")
    def __adjust_action__(self):
        
        target_x = float(self.target_info[u'x'])
        target_y = float(self.target_info[u'z'])
        agent_x = float(self.agent_info[u'x'])
        agent_y = float(self.agent_info[u'z'])
        
        recent_arrow_pos = (float(self.recent_arrow_info[u'x']), float(self.recent_arrow_info[u'z']))
        adj_block_right = (float(self.recent_arrow_info[u'x']), float(self.recent_arrow_info[u'z']) - 1.0)
        adj_block_left = (float(self.recent_arrow_info[u'x']), float(self.recent_arrow_info[u'z']) + 1.0)
        adj_block_down = (float(self.recent_arrow_info[u'x']) + 1.0, float(self.recent_arrow_info[u'z']) )
        adj_block_up = (float(self.recent_arrow_info[u'x']) - 1.0, float(self.recent_arrow_info[u'z']) )
        
        dis_r = math.sqrt( ( pow((target_x - adj_block_right[0]),2) + pow((target_y - adj_block_right[1]),2) ) ) 
        dis_l = math.sqrt( ( pow((target_x - adj_block_left[0]),2) + pow((target_y - adj_block_left[1]),2) ) ) 
        dis_up = math.sqrt( ( pow((target_x - adj_block_up[0]),2) + pow((target_y - adj_block_up[1]),2) ) ) 
        dis_down = math.sqrt( ( pow((target_x - adj_block_down[0]),2) + pow((target_y - adj_block_down[1]),2) ) )
        arrow_to_target = math.sqrt( ( pow((target_x - recent_arrow_pos[0]),2) + pow((target_y - recent_arrow_pos[1]),2) ) ) 
        
        blocks = {"right":dis_r, "left":dis_l, "up": dis_up, "down": dis_down}
        close_block = min(blocks, key=blocks.get)
        return [close_block,arrow_to_target]
        
                                
    def make_action(self, world_state,r):
        self.current_state = (int(self.target_info[u'x']),int(self.target_info[u'z']))
        v = []
        if self.current_state not in self.Qtable.keys():

            start = self.yaw_range[0]
            end = self.yaw_range[1]
            for i in range(start,end+1):
                for j in range(self.pitch_range[0],self.pitch_range[1]+1):
                    self.Qtable[self.current_state]["{}:{}".format(i,j)] = 0
                
            
                
        if self.prev_s is not None and self.prev_degree is not None:
            self.updateQTable(r)
            
        if self.first_action or (random.uniform(0,1) < epsilon):         
            v.append(random.randint(self.yaw_range[0],self.yaw_range[1]))
            v.append(random.randint(self.pitch_range[0],self.pitch_range[1]))
 
        else: 
            if self.prev_s == None and self.prev_degree == None:
                lis = [key for key in self.Qtable[self.current_state].keys() if self.Qtable[self.current_state][key] == max(self.Qtable[self.current_state].values())]
            else:
                lis = [key for key in self.Qtable[self.prev_s].keys() if self.Qtable[self.prev_s][key] == max(self.Qtable[self.prev_s].values())]
            best_action = np.random.choice(lis)
            l = best_action.split(":")
            print("best_action",best_action)
            v.append(l[0])
            v.append(l[1])
    
#             if self.target_info != None and self.recent_arrow_info != None:
#                 close_block,dis = self.__adjust_action__()
#                 if self.action == "yaw" and close_block in ["left","right"]:
#                     if close_block == "left" and (v-2) >= self.yaw_range[0]:
#                         v -= 2
#                     elif close_block == "right" and (v+2) <= self.yaw_range[1]:
#                         v += 2
#                 elif self.action == "pitch" and close_block in ["up","down"]:
#                     if close_block == "up" and (v+1) <= self.pitch_range[1]:
#                         v += 1
#                     elif close_block == "down" and (v-1) >= self.pitch_range[0]:
#                         v -= 1
                    
      
        agent_host.sendCommand("setYaw {}".format(v[0]))
        agent_host.sendCommand("setPitch {}".format(v[1]))

        time.sleep(0.1)
        agent_host.sendCommand("use 1")
        time.sleep(1)
        agent_host.sendCommand("use 0")
        
        self.prev_s = self.current_state
        self.prev_degree = "{}:{}".format(v[0],v[1])
            
    def get_recent_obs(self):
        
        while True:
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                obs_text = world_state.observations[-1].text
                obs = json.loads(obs_text) # most recent observation
                self.logger.debug(obs)
                for info in obs[u'entities']:
                    if info[u'name'] == u'Villager':
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
                    #print()
                    #print("Recent {}:({}, {})".format(self.recent_arrow_info[u'name'],self.recent_arrow_info[u'x'],self.recent_arrow_info[u'z']))
                    #print()
                break
           
        
        return world_state
    

    def get_dis(self):
        dis = 0
        if None not in (self.recent_arrow_info,self.target_info): 
            arrow_x = int(self.recent_arrow_info[u'x'])
            arrow_y = int(self.recent_arrow_info[u'z'])
            target_x = int(self.target_info[u'x'])
            target_y = int(self.target_info[u'z'])
            dis = math.sqrt( ( pow((target_x - arrow_x),2) + pow((target_y - arrow_y),2) ) )
            dis = round(dis,2)
        return dis
    
    def run(self, agent_host):
        total_discounted_reward = 0
        total_step = 30
        current_r = 0
        world_state = agent_host.getWorldState()
        global num_of_hits
        while world_state.is_mission_running:  
            for i in range(total_step):
                self.shots += 1
                if self.first_action:
                    while True:
                        time.sleep(0.1)
                        world_state = agent_host.getWorldState()
                        for error in world_state.errors:
                            self.logger.error("Error: %s" % error.text)
                            
                        if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                            world_state = self.get_recent_obs()
                            if self.target_info != None:
                                time.sleep(0.1)
                                self.make_action(world_state,current_r)
                            time.sleep(0.1)
                            world_state = self.get_recent_obs()
                            #print("current target health:",int(self.target_info[u'life']))
                            #print("self.target_health",self.target_health)
                            if self.target_info == None:
                                current_r = 10
                            elif int(self.target_info[u'life']) < self.target_health:
                                current_r = 10
                                self.target_health = int(self.target_info[u'life'])
                            else:
                                current_r = -1 * round(self.get_dis(),2)
                                
                            if self.first_action:
                                self.first_state = self.current_state
                                self.first_action = False    
                            break
                                
                else:              
                    while True: 
                        time.sleep(0.1)
                        world_state = agent_host.getWorldState()
                        for error in world_state.errors:
                            self.logger.error("Error: %s" % error.text)
                        
                        if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                            world_state = self.get_recent_obs()
                            if self.target_info != None:
                                time.sleep(0.1)
                                self.make_action(world_state,current_r)         
                            world_state = self.get_recent_obs()
                            if self.target_info == None:
                                current_r = 10
                            elif int(self.target_info[u'life']) < self.target_health:
                                current_r = 10
                                self.target_health = int(self.target_info[u'life'])
                            else:
                                current_r = -1 * round(self.get_dis(),2)
                            break
                        
                if current_r > 0:
                    num_of_hits += 1
                    print("current reward:",current_r)
                    print("num_of_hits",num_of_hits)
                    
                if self.target_info == None:
                    if self.prev_s != None and self.current_state != None:
                        total_discounted_reward += (gamma ** i) * self.updateQTable(current_r)
                    return(self.shots,total_discounted_reward)
                
            if self.prev_s != None and self.current_state != None:
                total_discounted_reward += (gamma ** i) * self.updateQTable(current_r)
                    
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

def drawfence(x1,z1,x2,z2,y):
    ''' Draw a powered rail between the two points '''
    y1railsegment = '" y1="' + str(y) + '" z1="'
    y2railsegment = '" y2="' + str(y) + '" z2="'
    y1stonesegment = '" y1="' + str(y-1) + '" z1="'
    y2stonesegment = '" y2="' + str(y-1) + '" z2="'
    shape = "north_south"
    if z1 == z2:
        shape = "east_west"
    rail_line = '<DrawLine x1="' + str(x1) + y1railsegment + str(z1) + '" x2="' + str(x2) + y2railsegment + str(z2) + '" type="golden_rail" variant="' + shape + '"/>'
    redstone_line = '<DrawLine x1="' + str(x1) + y1stonesegment + str(z1) + '" x2="' + str(x2) + y2stonesegment + str(z2) + '" type="fence" />'
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
    #print(drawrailline(-radius, 1-radius, -radius, radius-1, y))
    return drawrailline(-radius, 1-radius, -radius, radius-1, y) + \
           drawrailline(1-radius, radius, radius-1, radius, y) + \
           drawrailline(1-radius, -radius, radius-1, -radius, y) + \
           drawrailline(-radius, 1-radius, -radius, radius-1, y+1) + \
           drawrailline(1-radius, radius, radius-1, radius, y+1) + \
           drawrailline(1-radius, -radius, radius-1, -radius, y+1) + \
           drawblock(radius, y, radius) + drawblock(-radius, y, radius) + drawblock(-radius, y, -radius) + drawblock(radius, y, -radius)
  
def draw_fence(radius, y):          
    return drawfence(radius, 1-radius, radius, radius-1, y)
def random_pos():
    L = [(-3,0),(15,15),(15,-15),(30,0)]
    i = random.randint(0,len(L)-1)
    x,z = L[0]
    print("target pos: ({}, {})".format(x,z))
    return '<DrawEntity x="'+str(x)+'" y="57" z="' + str(z) + '" type="Villager" />'
    
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
                      <DrawCuboid type="air" x1="-9" x2="9" y1="57" y2="59" z1="9" z2="-9"/>
                      '''+draw_fence(5, 58)+'''
                      '''+drawloop(5, 58)+'''
                      '''+drawloop(5, 59)+'''
                      '''+random_pos()+'''
                      <DrawBlock x="15" y="57" z="0" type="redstone_block"/>
                      <DrawBlock x="15" y="57" z="1" type="redstone_block"/>
                      <DrawBlock x="14" y="57" z="0" type="redstone_block"/>
              
                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="300000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>MalmoTutorialBot</Name>
                <AgentStart>
                    <Placement x="15" y="58" z="0" yaw="75"/>
                    <Inventory>
                        <InventoryItem slot="0" type="bow"/>
                        <InventoryItem slot="1" type="arrow" quantity="60"/>
                    </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromNearbyEntities>
                     <Range name="entities" xrange="60" yrange="40" zrange="60"/>
                  </ObservationFromNearbyEntities>
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
#my_mission = MalmoPython.MissionSpec(missionXML, True)
#my_mission_record = MalmoPython.MissionRecordSpec()

agent = ArcherEnv(65,110,1,7)
num_of_repeats = 600
cumulative_rewards = []
result = {'0-199':0,'200-399':0,'400-599':0}
total_shots = 0
# Attempt to start a mission:
max_retries = 3
target_dead = False
for i in range(num_of_repeats):

    for retry in range(max_retries):
        try:
            #if i == 0 or target_dead:
            agent_host.startMission( my_mission, my_mission_record )
            #else:
            #    agent_host.startMission( my_mission_2, my_mission_record )
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
    
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            obs_text = world_state.observations[-1].text
            obs = json.loads(obs_text) # most recent observation
            for info in obs[u'entities']:
                if info[u'name'] == u'Villager':
                    target_info = info
                    break
            break

    #print("target_info",target_info) 
    agent.get_target_info(target_info)
    eta = max(min_lr,lr * (0.85 ** (i//100)))
    num_of_shots,cumulative_reward = agent.run(agent_host)
    total_shots += num_of_shots
    print("total shots:",total_shots)
    print("num of hits",num_of_hits)
    if total_shots != 0:
        print('Accurancy: {}%'.format(round((num_of_hits/total_shots),3) * 100))
        
    
    if i <= 199:
        epsilon -= 0.00125
        result['0-199'] = round((num_of_hits/total_shots),3) * 100
    elif i >= 200 and i <= 399:
        result['200-399'] = round((num_of_hits/total_shots),3) * 100
    elif i > 399:
        result['400-599'] = round((num_of_hits/total_shots),3) * 100
    print("epsilon=",epsilon)
    print('Cumulative reward: %d' % cumulative_reward)
    cumulative_rewards += [ cumulative_reward ]
    agent.reset()
    
    
    if i == 200 or i == 400:
        epsilon = 0.1
        total_shots = 0
        num_of_hits = 0
     
    time.sleep(0.1)
    agent_host.sendCommand("quit")
    time.sleep(1)

#print("total episodes reward:",cumulative_rewards)
#print()
for k,v in result.items():
    print("At game {} -> accuracy: {}%".format(k,v))

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