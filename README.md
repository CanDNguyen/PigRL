# PigRL
Malmo platform, is to train an agent to be proficient with Minecraft’s bow and arrow. The basic setup/ input will be to place the agent in a level environment with a bow and several arrows in their inventory, and the only other actor being a moving pig. The agent will be trained to kill the pig with the bow and arrow, which is to say that action will serve as the output.

## Milestone 1
The Goal of Milestone 1 is:<br>
* Create the Environment, the Pig(our target), the fences, the agent.<br>
* Create the "Random" Policy and "Greedy" policy for the agent to hit the area in the fences.<br>

How to run our Milestone 1:<br>
* clone PigRL repository
* On Terminal: <br>
   cd MalmoPlatform/Minecraft/<br>
   ./launchClient.sh (Mac)<br>
   launchClient.bat (Win)<br>
   cd MalmoPlatform/Python_Examples/<br>
   python3 Milestone1.py<br>
   

## Our Milestone 1 Walkthough GIF

<img src="http://g.recordit.co/Y92Lp7Sx3i.gif" width=450><br>


## Class ArcherEnv <An environment class> ##

Inside __init__:
   we define our actions as: "left-up","left-down","right-up","right-down","power"
   Initialize a Qtable for storing rewards.
   we store the agent's states and actions into dictionaries 
   we will develop value functions for the later milestone
   
Inside updateQTable:
   Passing the current reward and obseravtion to calculate the reward of each state
   Updating the QTable. 
   Our reward function is calculated based on distance formula.
   
Inside __convert_degree__:
   this method will recevie degrees and power that are chosen by the agent.
   It converts these numbers and used in time.sleep() function to control how long each command 
   will be exectuing.

Inisde make_action:
   this function receive the observation and apply policy on the agent's behavior of selecting actions.
   
turn_up_down, turn_left_right, and execute_action:
   execute actions.
   
Inside run:
   this function will start running the mission and assign actions to the agent.

