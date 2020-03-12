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
    

gamma = 0.4
lr = 1.0
min_lr = 0.002
epsilon = 0.6
max_vertical_degree = 20
max_horizontal_degree = 180
max_power = 10
eta = 0.3
target_info = None
num_of_hits = 0
arrow = 0
sword = 0

class ArcherEnv(object):
    
    def __init__(self,y1,y2,p):
        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.first_action = True
        self.first_state = None
        self.action_selection = ["bow","sword"]
        self.action = ""
        self.states = defaultdict(lambda: defaultdict(float))
        self.prev_degree = None
        self.recent_arrow_info = None
        self.target_info = None
        self.cart = None
        self.prev_s = None
        self.current_state = None
        self.shots = 0
        self.target_health = None
        self.Qtable = defaultdict(lambda: defaultdict(float))
        self.yaw_range = (y1,y2)
        self.pitch = p
        self.train_mode = True
        
    def get_target_info(self,t):
        self.target_info = t
        self.target_health = 10.0
    
    def evaluate(self):
        print("evaluation")
        self.train_mode = False
        states = None
        with open('result_1.txt') as inputtext:
            for line in inputtext:
                s = line.strip()
                #print("Line:",s)
                if s.startswith("#"):
                    states = s[2:-1].split(",")
                    states = (int(states[0]),int(states[1]))
                    
                else:
                    s = s.split(":")
                    #print(s)
                    self.Qtable[states]["{}:{}".format(s[0],s[1])] = float(s[2])
      
                
    def output_Q_table(self):
        with open('result4.txt','w') as f:
            for states,actions in self.Qtable.items():
                f.write("#{}\n".format(states))
                for a,v in actions.items():
                    f.write("{}:{}\n".format(a,v))
    def reset(self):
        self.current_state = None
        self.shots = 0
        self.prev_s = None
        self.prev_degree = None
        
    def updateQTable(self,reward):
        #for k,v in self.Qtable[self.prev_s].items():
        #    print("{} -> {}".format(k,v))
        print("\nraw reward for state:",reward)
        #print(self.Qtable[self.current_state].values())
        
        print("updating",self.prev_s)
        #print("degree",self.prev_degree)
  
        
        self.Qtable[self.prev_s][self.prev_degree] = self.Qtable[self.prev_s][self.prev_degree] + eta * \
                                                    (reward + gamma * max(self.Qtable[self.current_state].values()) - self.Qtable[self.prev_s][self.prev_degree])
        #print("{} + {} * ({} + {} * {}) - {} = {}".format(self.Qtable[self.prev_s][self.prev_degree], eta, reward,gamma,max(self.Qtable[self.current_state].values()),self.Qtable[self.prev_s][self.prev_degree],self.Qtable[self.prev_s][self.prev_degree]))                                            
        print("{} at state: {}: {}".format(self.prev_degree, self.prev_s,self.Qtable[self.prev_s][self.prev_degree]))
        #self.Qtable[self.prev_s][self.prev_degree] = (1-eta) * self.Qtable[self.prev_s][self.prev_degree] + eta * \
        #                                            (reward + gamma * max(self.Qtable[self.current_state].values()))
                                                    
        #self.__debuginfo__()
        return reward
            
        #print("Qtable:",self.Qtable)
    
    def __debuginfo__(self):
        
        print("************************************************")
        
        print("Total states:",len(self.Qtable))
        print("actions in current state that not explored",len([k for k,v in self.Qtable[self.current_state].items() if v == 0]))
#         print("\nQtable:\n")
#         for key in self.Qtable.keys():
#             print("state: ",key)    
#             for k,v in self.Qtable[key].items():
#                 print("{} -> {}".format(k,v))
#             print("-------------------------------------------")
        
        
        print("************************************************")
        
    def get_distance(self):
        target_x = int(self.target_info[u'x'])
        target_y = int(self.target_info[u'z'])
        agent_x = 0
        agent_y = 0
        dis = math.sqrt( ( pow((target_x - agent_x),2) + pow((target_y - agent_y),2) ) )    
        return dis
    
    def make_action(self, world_state,r):
        
        global arrow
        global sword
        self.current_state = (int(self.target_info[u'x']),int(self.target_info[u'z']))
        
            
        print("current",self.current_state)
        print("previous:",self.prev_s)
        v = []
        self.action = ""
        d = self.get_distance()
        if self.current_state not in self.Qtable.keys():
            start = self.yaw_range[0]
            end = self.yaw_range[1]
            for i in range(start,end+1,5):
                #for j in range(self.pitch_range[0],self.pitch_range[1]+1):
                self.Qtable[self.current_state]["{}:{}:{}".format(i,5,"bow")] = 0
                self.states[self.current_state]["{}:{}:{}".format(i,5,"bow")] = 0
            for i in range(start,end+1,15):
                self.Qtable[self.current_state]["{}:{}:{}".format(i,28,"sword")] = 0
                self.states[self.current_state]["{}:{}:{}".format(i,28,"sword")] = 0
                
            self.states[self.current_state]['max'] = 0
        
                        
        if self.prev_s is not None and self.prev_degree is not None and self.train_mode:
            self.updateQTable(r)
            
        if self.first_action or (random.uniform(0,1) < epsilon):                 
            explore = [k for k,v in self.Qtable[self.current_state].items() if v == 0]
            if len(explore) > 0:
                a = np.random.choice(explore)           
            else:
                a = np.random.choice(list(self.Qtable[self.current_state]))
            a = a.split(":")
            v.append(int(a[0]))
            v.append(int(a[1]))
            self.action = a[2]
            print(v[0],":",v[1])   
            
        else: 
            lis = [key for key in self.Qtable[self.current_state].keys() if self.Qtable[self.current_state][key] == max(self.Qtable[self.current_state].values())]
            best_action = np.random.choice(lis)
            print("best action",best_action)
            l = best_action.split(":")
            #print("best_action",best_action)
            v.append(int(l[0]))
            v.append(int(l[1]))
            self.action = l[2]


            
        if self.action == "bow":
            arrow += 1
            agent_host.sendCommand("setYaw {}".format(v[0]))
            agent_host.sendCommand("setPitch {}".format(v[1]))
            agent_host.sendCommand("hotbar.1 1")
            agent_host.sendCommand("hotbar.1 0")
            agent_host.sendCommand("use 1")
            time.sleep(1)
            agent_host.sendCommand("use 0")
        elif self.action == "sword":
            print("using sword")
            sword += 1
            agent_host.sendCommand("setYaw {}".format(v[0]))
            agent_host.sendCommand("hotbar.3 1")
            agent_host.sendCommand("hotbar.3 0")
            for i in range(v[1],68,5):
                agent_host.sendCommand("setPitch {}".format(i))
                agent_host.sendCommand("attack 1")
                time.sleep(0.1)
                agent_host.sendCommand("attack 0")
        
        self.prev_s = self.current_state
        self.prev_degree = "{}:{}:{}".format(v[0],v[1],self.action)
        
          
    def get_recent_obs(self):
        
        last_pos = []
        while True:
            self.target_info = None
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                obs_text = world_state.observations[-1].text
                obs = json.loads(obs_text) # most recent observation
                self.logger.debug(obs)
                #for i in world_state.observations:
                #    obs = json.loads(i.text)                
                    
                for info in obs[u'entities']:
                    if info[u'name'] == u'Pig':
                        print(info[u'x'],info[u'z'])       
                        self.target_info = info
                        break
                
                if obs[u'entities'][-1]['name'] == u'Arrow':
                    self.recent_arrow_info = obs[u'entities'][-1]
                
                if self.target_info == None:
                    return world_state
                
                if last_pos != []:
                    if self.target_info != None and self.target_info[u'x'] == last_pos[0] and self.target_info[u'z'] == last_pos[1]:
                        return world_state
                    else:
                        last_pos = []
                last_pos.append(self.target_info[u'x'])
                last_pos.append(self.target_info[u'z'])
                #if self.target_info != None and float(self.target_info[u'motionX']) == 0.0:
                #    return world_state
                #elif self.target_info == None:
                #    return world_state
                
    def get_recent_obs_1(self):
        
        while True:
            self.target_info = None
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                obs_text = world_state.observations[-1].text
                obs = json.loads(obs_text) # most recent observation
                self.logger.debug(obs)
                #for i in world_state.observations:
                #    obs = json.loads(i.text)
                for info in obs[u'entities']:
                    if info[u'name'] == u'Pig':
                        print(info[u'x'],info[u'z'])        
                        self.target_info = info
                        break
                
                if obs[u'entities'][-1]['name'] == u'Arrow':
                    self.recent_arrow_info = obs[u'entities'][-1]
                return world_state
            
    
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
                        if world_state.is_mission_running:    
                            #time.sleep(1.5)
                            world_state = self.get_recent_obs()
                            if self.target_info != None:
                                self.make_action(world_state,current_r)
                            time.sleep(0.5) 
                            world_state = self.get_recent_obs_1()
                            #print("current target health:",int(self.target_info[u'life']))
                            #print("self.target_health",self.target_health)
                            if self.first_action:
                                self.first_state = self.current_state
                                self.first_action = False    
                            break
                                
                else:              
                    while True:        
                        if world_state.is_mission_running: 
                            #time.sleep(1.5)
                            world_state = self.get_recent_obs()
                            if self.target_info != None:
                                self.make_action(world_state,current_r) 
                            time.sleep(0.5)       
                            world_state = self.get_recent_obs_1()   
                            break
                
                if self.target_info == None:
                    print("get reward from self.target_info == None")
                    if self.action == "sword":
                        current_r = 40
                    else:
                        current_r = 20

                elif int(self.target_info[u'life']) < self.target_health:
                    print("get reward from int(self.target_info[u'life']) < self.target_health")
                    if self.action == "sword":
                        current_r = 20
                    else:
                        current_r = 10
                    self.target_health = int(self.target_info[u'life'])
                else:
                    current_r = -5   
                    
                total_discounted_reward += (gamma ** i) * current_r
                self.__debuginfo__()
                if current_r > 0:
                    #self.states[self.prev_s][self.prev_degree] += 1
                    #self.states[self.prev_s]['max'] = max(self.states[self.prev_s].values()) 
                    num_of_hits += 1
                    print("current reward:",current_r)
                    print("num_of_hits",num_of_hits)
                
                if self.target_info == None and i < 29:
                    #if self.prev_s != None and self.current_state != None and self.train_mode:
                    #    total_discounted_reward += (gamma ** i) * self.updateQTable(current_r)
                    agent_host.sendCommand("quit")
                    time.sleep(1)
                    my_mission = MalmoPython.MissionSpec(getMyMission(), True)
                    agent_host.startMission( my_mission, my_mission_record )
                    self.target_health = 10.0
                    #return(self.shots,total_discounted_reward)
                
            if self.prev_s != None and self.current_state != None and self.train_mode:
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
    return '<DrawBlock x="' + str(x) + ystonesegment + str(z) + '" type="fence"/>'
def drawloop(radius, y):
    ''' Create a loop of powered rail '''
    #print(drawrailline(-radius, 1-radius, -radius, radius-1, y))
    return drawrailline(-radius, 1-radius, -radius, radius-1, y) + \
           drawrailline(1-radius, radius, radius-1, radius, y) + \
           drawrailline(1-radius, -radius, radius-1, -radius, y) + \
           drawblock(radius, y, radius) + drawblock(-radius, y, radius) + drawblock(-radius, y, -radius) + drawblock(radius, y, -radius)
  
def draw_fence(radius, y):          
    return drawfence(radius, 1-radius, radius, radius-1, y)

def get_pos():
    l = []
    for i in range(-7,9):
        for j in range(-7,9):
            s = (i,j)
            l.append(s)
    l.remove((0,0))
    return l
def random_pos(L):

    #L = [(-3,0),(15,15),(15,-15),(30,0)]
    i = random.randint(0,len(L)-1)
    x,z = L[i]
    print("target pos: ({}, {})".format(x,z))
    return '<DrawEntity x="'+str(x)+'" y="57" z="' + str(z) + '" type="Pig" />'

def output_stats(cr):
    with open('stats-4.txt','w') as f:
        for i in enumerate(cr):
            f.write("{}:{}\n".format(i[0],i[1]))
#Pos = read_position() 
Pos = get_pos()

def getMyMission():
    missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                
                  <About>
                    <Summary>Hello world!</Summary>
                  </About>
                  <ModSettings>
                    <MsPerTick>20</MsPerTick>
                  </ModSettings>
                  <ServerSection>
                    <ServerInitialConditions>
                      <Time>
                        <StartTime>12000</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime>
                      </Time>
                      <Weather>clear</Weather>
                    </ServerInitialConditions>
                    <ServerHandlers>
                      <FlatWorldGenerator generatorString="3;7,56*1:9,36;,biome_1"/>
                      <DrawingDecorator>
                          <DrawCuboid type="air" x1="-9" x2="9" y1="57" y2="59" z1="-9" z2="9"/>
                          '''+draw_fence(9, 58)+'''
                          '''+draw_fence(9,59)+'''
                          '''+draw_fence(9, 60)+'''
                          '''+drawloop(9, 58)+'''
                          '''+drawloop(9,59)+'''
                          '''+drawloop(9,60)+'''
                          '''+random_pos(Pos)+'''
                  
                      </DrawingDecorator>
                      <ServerQuitFromTimeUp timeLimitMs="300000"/>
                      <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                  </ServerSection>
                  
                  <AgentSection mode="Survival">
                    <Name>MalmoTutorialBot</Name>
                    <AgentStart>
                        <Placement x="0" y="58" z="0" yaw="-135"/>
                        <Inventory>
                            <InventoryItem slot="0" type="bow"/>
                            <InventoryItem slot="1" type="arrow" quantity="60"/>
                            <InventoryItem slot="2" type="stone_sword"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                      <InventoryCommands/>
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
    return missionXML
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

my_mission = MalmoPython.MissionSpec(getMyMission(), True)
my_mission_record = MalmoPython.MissionRecordSpec()
#my_mission = MalmoPython.MissionSpec(missionXML, True)
#my_mission_record = MalmoPython.MissionRecordSpec()
# 67,108,7,18
agent = ArcherEnv(0,360,5)
num_of_repeats = 900
cumulative_rewards = []
result = {'0-299':0,'300-599':0,'600-899':0}
total_shots = 0
# Attempt to start a mission:
max_retries = 3
hit_or_miss = []
#agent.evaluate()
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
    agent_host.sendCommand("setPitch {}".format(28))
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            obs_text = world_state.observations[-1].text
            obs = json.loads(obs_text) # most recent observation
            #agent_x = 0
            #agent_y = 0
            for info in obs[u'entities']:
                #if info[u'name'] == u'MalmoTutorialBot':
                    #agent_x  = int(info[u'x'])
                    #agent_y  = int(info[u'z'])
                    #print(info[u'yaw'],info[u'pitch'])
                    #print(info[u'x'],info[u'z'])
                    #break
                if info[u'name'] == u'Pig':
                   target_info = info
                   break
            #target_x = int(target_info[u'x'])
            #target_y = int(target_info[u'z'])
            #dis = math.sqrt( ( pow((target_x - agent_x),2) + pow((target_y - agent_y),2) ) )    
            #print(dis)
            #print(target_info[u'motionX'],target_info[u'motionY'])
#             for i in range(0,361,20):
#                 if world_state.number_of_observations_since_last_state > 0:
#                     obs_text = world_state.observations[-1].text
#                     obs = json.loads(obs_text)
#                     for info in obs[u'entities']:
#                         if info[u'name'] == u'MalmoTutorialBot':
#                             print(info[u'yaw'],info[u'pitch'])
#                 agent_host.sendCommand("setYaw {}".format(i))
#                 time.sleep(1)
#                 agent_host.sendCommand("hotbar.3 1")
#                 agent_host.sendCommand("hotbar.3 0")
#                 for j in range(28,68,10):
#                     agent_host.sendCommand("setPitch {}".format(j))
#                     agent_host.sendCommand("attack 1")
#                     time.sleep(0.5)
#                     agent_host.sendCommand("attack 0")
#                 world_state = agent_host.getWorldState()
            break

    #print("target_info",target_info) 
    agent.get_target_info(target_info)
    #eta = max(min_lr,lr * (0.85 ** (i//100)))
    num_of_shots,cumulative_reward = agent.run(agent_host)
    total_shots += num_of_shots
    print("total shots:",total_shots)
    print("num of hits",num_of_hits)
    if total_shots != 0:
        print('Accurancy: {}%'.format(round((num_of_hits/total_shots),3) * 100))
        
    
    if i == 299:
        result['0-299'] = round((num_of_hits/total_shots),3) * 100
    elif i == 599:
        result['300-599'] = round((num_of_hits/total_shots),3) * 100
    elif i == 899:
        result['600-899'] = round((num_of_hits/total_shots),3) * 100
    print("epsilon=",epsilon)
    print('Cumulative reward: %d' % cumulative_reward)
    cumulative_rewards += [ cumulative_reward ]
    agent.reset()
    
    if i == 299:
        epsilon = 0.4
        hit_or_miss.append((num_of_hits,total_shots))
        total_shots = 0
        num_of_hits = 0
    elif i == 599:
        epsilon = 0.2
        hit_or_miss.append((num_of_hits,total_shots))
        total_shots = 0
        num_of_hits = 0
     
    time.sleep(0.1)
    agent_host.sendCommand("quit")
    time.sleep(1)
    my_mission = MalmoPython.MissionSpec(getMyMission(), True)

hit_or_miss.append((num_of_hits,total_shots))
output_stats(cumulative_rewards)
agent.output_Q_table()
for k,v in result.items():
    print("At game {} -> accuracy: {}%".format(k,v))
if len(hit_or_miss) > 0:
    print("action 0-9000:\nhit:{}\nmiss:{}\ntotal:{}".format(hit_or_miss[0][0],hit_or_miss[0][1]-hit_or_miss[0][0],hit_or_miss[0][1]))
    print("action 9000-18000:\nhit:{}\nmiss:{}\ntotal:{}".format(hit_or_miss[1][0],hit_or_miss[1][1]-hit_or_miss[1][0],hit_or_miss[1][1]))
    print("action 18000-27000:\nhit:{}\nmiss:{}\ntotal:{}".format(hit_or_miss[2][0],hit_or_miss[2][1]-hit_or_miss[2][0],hit_or_miss[2][1]))
print("using sword",sword)
print("using arrow",arrow)
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