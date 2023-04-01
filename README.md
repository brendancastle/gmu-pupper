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
    * `pip3 install numpy transforms3d pyserial ray arspb`
    * inside puppersim repo: `pip3 install -e .`
4. run the simulator server: `python3 puppersim/pupper_server.py`
5. run the pupper on simulator server: `python3 run_djipupper_sim.py`
6. you can use keyboard to control it:
    * wasd --> moves robot forward/back and left/right
    * arrow keys --> turns robot left/right
    * q --> activates/deactivates robot
    * e --> starts/stops trotting gait
    * ijkl --> tilts and raises robot

NB. it may be that the pupper doesnt work. thats because we're working on it to add functionalities so it may cause pupper to crash :)

