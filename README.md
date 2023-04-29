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
2. press `e` to start trotting and position robot wherever
3. when ready press `t` to start "follow mode". this makes the robot stand so it will re-orient itself
4. press `u` to make robot start following
5. press `u` again to make it stop following and make it stand
6. press `t` again to exit "follow mode"
7. now you can manually take over the robot by pressing `e`

### UPDATE:
1. pupper is going to go to a clone in the simulator
2. I've made changes to the simulator server but it should work on any server with a cone

### parameters which we can tune:
* FollowController.py:
    1. self.l_alpha: Its like a slow up/ slow down in the forward/backward direction. Default value: 0.15
    2. self.r_alpha: Similar as above but for the yaw value. Default: 0.1
    3. self.eps: The distance at which pupper stops. Default: 0.5
    4. self.slow_down_distance: Deprecated. Pupper starts slowing down at this distance


NB. it may be that the pupper doesnt work. thats because we're working on it to add functionalities so it may cause pupper to crash :)

