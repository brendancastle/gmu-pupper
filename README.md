# gmu-pupper
George Mason University CS 685 Robotics Project

## Team Roster
- Brendan Castle
- Saad M A Ghani
- Jared DiCioccio
- Archange Giscard Destiné

## DJI Branch
This branch is for simulations. It is very similar to the main code but has different file names and slightly different logic due to the use of keyboard vs the controller used by pupper.

### running simulator on ubuntu 20.04.6:
1. clone the puppersim repo (seperately) `git clone https://github.com/jietan/puppersim/`
2. (optional) create conda environment. puppersim required python 3.7. `conda create --name rl_pupper python=3.7`
3. install dependencies: 
    * `pip3 install numpy transforms3d pyserial ray arspb quadprog`
    * inside puppersim repo: `pip3 install -e .`
4. run the simulator server: `python3 puppersim/pupper_server.py`
5. run the pupper on simulator server: `python3 run_djipupper_sim.py`
6. you can use keyboard to control it:
    * wasd --> moves robot forward/back and left/right
    * arrow keys --> turns robot left/right
    * q --> activates/deactivates robot
    * e --> starts/stops trotting gait
    * ijkl --> tilts and raises robot

## added functionality:
When running simulator:
1. press `q` to activate robot
2. robot automatically starts moving towards cone in view

if automated = false on line 35 of run_djipupper_sim.py, after activating:
2. press `e` to start trotting and position robot wherever
3. when ready press `t` to start "follow mode". robot automatically starts going towards cone in view
4. press `t` again to exit "follow mode"
5. now you can manually take over the robot by pressing `e`

### UPDATE:
1. pupper automatically going towards cone in view using camera
2. ive added a file "pupper_server_saad.py" which you can copy-paste into your pupper_server.py of puppersim repo. It is the modified server which has 1 cone

### parameters which we can tune:
* FollowController.py:
    1. self.l_alpha: Its like a slow up/ slow down in the forward/backward direction. Default value: 0.15
    2. self.r_alpha: Similar as above but for the yaw value. Default: 0.1
    3. self.eps: The distance at which pupper stops. Default: 0.5. Doesnt work - depth needs tuning
    

NB. it may be that the pupper doesnt work. thats because we're working on it to add functionalities so it may cause pupper to crash :)

