# robust-drone-pathfollowing 
Robust-drone-pathfollowing is a [Gym-Pybullet-Drones](https://github.com/utiasDSL/gym-pybullet-drones) extension, which is a [simple](https://en.wikipedia.org/wiki/KISS_principle) OpenAI [Gym environment](https://gym.openai.com/envs/#classic_control) based on [PyBullet](https://github.com/bulletphysics/bullet3) for multi-agent reinforcement learning with quadrocopter. This extension provides a Gym environment that uses a stochastical, random wind field for single-agent reinforcment learning.

> ## Citation
> will be added later...

## Installation
This repo was written using Python3.8 with conda on Ubuntu 20.04. However since it is only a [Gym-Pybullet-Drones](https://github.com/utiasDSL/gym-pybullet-drones) extension, it should be compatible with macOS 12 and also other future compatibilities.

### Conda
First install Conda. For Ubuntu 20.04 read [here](https://linuxize.com/post/how-to-install-anaconda-on-ubuntu-20-04/) for Information and Installation Instructions.

### Gym-Pybullet-Drones
More Information about [Gym-Pybullet-Drones](https://github.com/utiasDSL/gym-pybullet-drones) can be found on the corresponding Github repo.
```bash
$ conda create -n drones python=3.8
$ conda activate drones
$ pip3 install --upgrade pip
$ git clone https://github.com/utiasDSL/gym-pybullet-drones.git
$ cd gym-pybullet-drones/
$ pip3 install -e .
```

### Robust-Drone-Pathfollowing
Then, do:
```bash
$ git clone https://gitlab2.informatik.uni-wuerzburg.de/s408133/robust-drone-pathfollowing.git
```
Now, some files inside the changefiles directory have to be copied inside Gym-Pybullet-Drones:
```bash
$ cp robust_drone_pathfollowing/changefiles/WindSingleAgentAviary.py gym_pybullet_drones/envs
$ cp robust_drone_pathfollowing/changefiles/__init__.py gym_pybullet_drones/envs/__init__.py
$ cp robust_drone_pathfollowing/changefiles/other/__init__.py gym_pybullet_drones/__init__.py
```

### Test Installation
You can verify that there was no Problem by:
```bash
$ cd robust_drone_pathfollowing
$ python3 testInstall.py
```
Now you should see a flying drone that maybe is hit by a flying duck.
You can change different parameters, see:
```bash
$ python3 testInstall.py -h
```
By increasing the force vector of the wind you can see that the standard controller is not really robust against wind. 
It should be noted that in this script no generic wind environment was used. The wind force is appplied each time the controller is able to send RPMs in the script.
To really evaluate this controller the forces should be applied at each environment time step like done in the WindSingleAgentAviary class.
To further test the installation, use:
```bash
$ python3 learn.py
```
If done correctly, this should result in an short PPO learning process.

## HelpClasses
There are a couple of extensions that this extension is using. The most important classes will be shortly explained hereafter.

### Class `Wind`
The Wind class implements a random 3D wind field. It provides the necassary force vector dependent on the current position. In addition it can visualize the wind field like:

<img src="files/readme_images/Figure_1.png" alt="random wind field with vortexes" width="400"><img src="files/readme_images/Figure_2.png" alt="random wind field" width="400">

### Class `PathPlotter`
The PathPlotter class plots the path of the drone, as well as its goal. This resulst in plots like the one seen herafter(Agent not learned).

### Class `EvalWriter`
The EvalWriter class evaluates a model and writes its performance to an xlsx file. It evaluates the sucess rate, the sucess time rate, the average distance half way through the simulation, the average reward and plots path / goal if it is a signel evaluation by using the PathPlotter class.


<img src="files/readme_images/Figure_5.png" alt="drone path and goal" width="400"><img src="files/readme_images/Figure_7.png" alt="distances to goal" width="400">

## Class `WindSingleAgentAviary`
The WindSingleAgentAviary class is a subclass of the [SingleAgentAviary] (https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/single_agent_rl/BaseSingleAgentAviary.py) class. It models the Single Agent Problem to hover at a position under influence of strong wind.

```python
>>> env = WindSingleAgentAviary( 
>>>       drone_model=DroneModel("hb"),     # quadcopter that weighs 500g 
>>>       initial_xyzs=None,                # Initial XYZ positions of the drones
>>>       initial_rpys=None,                # Initial roll, pitch, and yaw of the drones in radians 
>>>       physics: Physics=Physics.PYB,     # Choice of (PyBullet) physics implementation 
>>>       freq=240,                         # Stepping frequency of the simulation
>>>       aggregate_phy_steps=1,            # Number of physics updates within each call to BaseAviary.step()
>>>       gui=False,                        # Whether to display PyBullet's GUI
>>>       record=False,                     # Whether to save a video
>>>       obs=ObservationType.KIN,          # The observation type
>>>       act=ActionType.RPM                # The action type
>>>       total_force=0.00                  # The total force of the Wind (N)
>>>       mode=0                            # The mode of the environment
>>>       episode_len=5                     # The amount of seconds of each episode
>>>       upper_bound=1.0                   # The upperbound of where the goal can be in each axis
>>>       debug=False)                      # Whether to use debug lines
````


The environment posses some different modes:

| mode | Wind | Goal |
|---------------------------------: | :-------------------: | :-------------------------------------------: |
| 0 | no wind | (0,0,z) |
| 1 | no wind | (x,y,z) |
| 2 |   yes, constant   | (x,y,z) |
| 3 |   yes, random     | (x,y,z) |

The environment can be instantiated by using `gym.make()`—see [`learn.py`](https://github.com/eislerMeinName/robust_drone_pathfollowing/blob/main/learn.py) for an example.

```python
>>> env = gym.make('WindSingleAgent-aviary-v0')
```
The environment can be stepped for example with an easy for loop:

```python
>>> obs = env.reset()
>>> for _ in range(10*240):
>>>     obs, reward, done, info = env.step(env.action_space.sample())
>>>     env.render()
>>> env.close()
```

## Script `learn.py`
This Script should be used to learn your single agent based on PPO or DDPG Algorithm. You can either create a new model or load a model that already exists. Then the model is trained with the chosen amount of steps and saved. The script has the following optional arguments and can be executed with `python3 learn.py` inside your conda environment:
```
  -h, --help      show this help message and exit
  --algo          The Algorithm that trains the agent(PPO(default), DDPG)
  --obs           The chosen ObservationType (default: KIN)
  --act           The chosen ActionType (default: RPM)
  --cpu           Amount of parallel training environments (default: 1)
  --steps         Amount of training time steps(default: 100000)
  --folder        Output folder (default: results)
  --mode          The mode of the training environment(default: 0)
  --env           Name of the environment(default:WindSingleAgent-aviary-v0)
  --load          Load an existing model(default: False)
  --load_file     The experiment folder where the loaded model can be found
  --total_force   The max force in the simulated Wind field(default: 0)
  --upper_bound   The upper bound of the area where the goal is simulated(default: 1)
  --debug_env     Parameter to the Environment that enables most of the Debug messages(default: False)
  --episode_len   The episode length(default: 5)
```

## Script `eval.py`
This Script should be used to evaluate your single agent based on PPO or DDPG Algorithm. It creates an evaluation environment and loads the model. Then it evaluates the model with the EvalWriter class. If choosen it also creates a new test environment, a logger and a pathplotter to test a single performance with visible output in the GUI. The script has the following optional arguments and can be executed with `python3 learn.py` inside your conda environment:

```
  -h, --help           show this help message and exit
  --algo               The Algorithm that trains the agent(PPO(default), DDPG)
  --obs                The chosen ObservationType (default: KIN)
  --act                The chosen ActionType (default: RPM)
  --folder             Output folder (default: results)
  --mode               The mode of the training environment(default: 0)
  --episodes           The number of evaluation steps(default: 100)
  --env                Name of the environment(default:WindSingleAgent-aviary-v0)
  --load_file          The experiment folder where the loaded model can be found
  --total_force        The max force in the simulated Wind field(default: 0)
  --upper_bound        The upper bound of the area where the goal is simulated(default: 1)
  --debug_env          Parameter to the Environment that enables most of the Debug messages(default: False)
  --gui                Enables/ Disables the gui replay(default: True)
  --gui_time GUI_TIME  The simulation length(default: 10)

```

> ## References
> will be added later...

