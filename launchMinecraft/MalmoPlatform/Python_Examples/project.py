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


if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)
    
Policy = ["random","greedy"]

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
        
        self.policy = Policy[0]
        
        self.action_selection = ["left-up","left-down","right-up","right-down","power"]
        self.action = ""
        self.states = {} #to do
        self.prev_s = None
        self.prev_a = None
        self.current_state = None
        
        self.trajectory = {} # state-action-reward sequence
        
        self.Qtable = {}

        self.state_space = {} # a table to store state-value
        self.action_space = {} # store action-value
        
        self.trans = [] # transition probability distribution
    
    def updateQTable(self,reward,world_state):
        
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text)
        recent_arrow_info = None
        target_info = None
        #get most recent arrow's pos
        
        print("this is OBS: ",obs)
        print("obs['entities'][-1]['name']: ",obs['entities'][-1]['name'])
        if obs[u'entities'][-1][u'name'] == u'Arrow':
            recent_arrow_info = obs[u'entities'][-1]
            print("recent_arrow_info: ",recent_arrow_info)
            print("{}:({}, {})".format(recent_arrow_info[u'name'],recent_arrow_info[u'x'],recent_arrow_info[u'z']))
            
        for info in obs[u'entities']:
            if info[u'name'] == u'Arrow':
                print("recent_arrow_info: ",info)
                recent_arrow_info = info
     
        for info in obs[u'entities']:
            if info[u'name'] == u'Pig':
                target_info = info
                #print("{}:({}, {})".format(target_info[u'name'],target_info[u'x'],target_info[u'z']))
                break;
        print("Arrow",recent_arrow_info)
        print("Pig",target_info)
        
        if None not in (recent_arrow_info,target_info):
            print("hehehe")
            arrow_x = int(recent_arrow_info[u'x'])
            arrow_y = int(recent_arrow_info[u'z'])
            target_x = int(target_info[u'x'])
            target_y = int(target_info[u'z'])
            dis = math.sqrt( ( pow((target_x - arrow_x),2) + pow((target_y - arrow_y),2) ) )
            print("distance:",dis)
        
        print("hahaha",reward)
     
        if reward == 0:
            self.Qtable[self.current_state] += -1 * dis
        else:
            self.Qtable[self.current_state] += reward
            
        
            
        #print("Qtable:",self.Qtable)
                
    def __convert_degree__(self,L):
        
        to_return = {}
        for k,v in L.items():
            if k == "power":
                to_return[k] = v / 10
            else:
                to_return[k] = v * 0.005
        return to_return
        
    def make_action(self, world_state):
        
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text) # most recent observation
        self.logger.debug(obs)
        
        res = []
        # random version
        if self.policy == "random":
            random_action = random.randint(0,(len(self.action_selection))-2)
            print("select {}".format(self.action_selection[random_action]))
#             if self.action_selection[random_action] == "left-up":
#                 random_vertical_degree = random.randint(0,90)
#                 random_horizontal_degree = random.randint(-90,0)
#                 random_power = random.randint(2,10)
#
#             elif self.action_selection[random_action] == "left-down":
#                 random_vertical_degree = random.randint(-90,0)
#                 random_horizontal_degree = random.randint(-90,0)
#                 random_power = random.randint(2,10)
#
#             elif self.action_selection[random_action] == "right-up":
#                 random_vertical_degree = random.randint(0,90)
#                 random_horizontal_degree = random.randint(0,90)
#                 random_power = random.randint(2,10)
#
#             elif self.action_selection[random_action] == "right-down":
#                 random_vertical_degree = random.randint(-90,0)
#                 random_horizontal_degree = random.randint(0,90)
#                 random_power = random.randint(2,10)

            random_vertical_degree = random.randint(0,90)
            random_horizontal_degree = random.randint(0,180)
            random_power = random.randint(2,10)
            self.action = self.action_selection[random_action]
            self.current_state = "{}:{}:{}".format(random_vertical_degree,random_horizontal_degree,random_power)
            print("key:",self.current_state)
            self.states[self.current_state] = 0
            self.Qtable[self.current_state] = 0
            
            res = self.__convert_degree__({"vertical_degree":random_vertical_degree,
                                          "horizontal_degree":random_horizontal_degree,
                                          "power":random_power})
            
        elif self.policy == "greedy":
            # to do: get the closest arrow's position and check its surrounding block (3x3)
            #        find the block that has the closest distance to the target
            #        select that block as our next state
            best_pos = np.argmin(self.Qtable)
            print(best_pos)
        return res
    
    def execute_action(self,agent_host,commands):
        print("command",commands)

        if self.action == "left-up":
            agent_host.sendCommand("turn -1")
            time.sleep(commands["vertical_degree"])
            agent_host.sendCommand("turn 0")
            agent_host.sendCommand("pitch -1")
            time.sleep(commands["horizontal_degree"])
            agent_host.sendCommand("pitch 0")
        elif self.action == "left-down":
            agent_host.sendCommand("turn -1")
            time.sleep(commands["vertical_degree"])
            agent_host.sendCommand("turn 0")
            agent_host.sendCommand("pitch 1")
            time.sleep(commands["horizontal_degree"])
            agent_host.sendCommand("pitch 0")
        elif self.action == "right-up":
            agent_host.sendCommand("turn 1")
            time.sleep(commands["vertical_degree"])
            agent_host.sendCommand("turn 0")
            agent_host.sendCommand("pitch -1")
            time.sleep(commands["horizontal_degree"])
            agent_host.sendCommand("pitch 0")
        else:
            agent_host.sendCommand("turn 1")
            time.sleep(commands["vertical_degree"])
            agent_host.sendCommand("turn 0")
            agent_host.sendCommand("pitch 1")
            time.sleep(commands["horizontal_degree"])
            agent_host.sendCommand("pitch 0")
        
        time.sleep(0.2)
        agent_host.sendCommand("use 1")
        time.sleep(commands["power"])
        agent_host.sendCommand("use 0")
        time.sleep(0.2)
        
    def run(self, agent_host):
        
        total_reward = 0
        
        world_state = agent_host.getWorldState()
        while world_state.is_mission_running:
            
            current_r = 0
            while True:
                time.sleep(0.1)
                world_state = agent_host.getWorldState()
                for error in world_state.errors:
                    self.logger.error("Error: %s" % error.text)
                for reward in world_state.rewards:
                    print("reward.getValue()",reward.getValue())
                    current_r += reward.getValue()
                if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                    self.execute_action(agent_host, self.make_action(world_state))
                    self.updateQTable(current_r, world_state)
                    self.prev_s = self.current_state
                    self.prev_a = self.action
                    
                    
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
                      <DrawEntity x="0" y="57" z="0" xVel="0" yVel="0" zVel="-1" type="MinecartRideable"/>
                      <DrawEntity x="0" y="57" z="0" type="Pig"/>
                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="10000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>MalmoTutorialBot</Name>
                <AgentStart>
                    <Placement x="15" y="57" z="0" yaw="90"/>
                    <Inventory>
                        <InventoryItem slot="0" type="bow"/>
                        <InventoryItem slot="1" type="arrow" quantity="10"/>
                    </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromNearbyEntities>
                     <Range name="entities" xrange="60" yrange="40" zrange="60"/>
                  </ObservationFromNearbyEntities>
                  <RewardForDamagingEntity>
                      <Mob type ="Pig" reward="10"/>
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
# Attempt to start a mission:
max_retries = 3
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


print("Mission running ", end=' ')

agent.run(agent_host)
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
