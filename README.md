# Multi Agent Systems: Perudo Project

This repository contains a simulation of the game Perudo. Probability theory and epistemic logic are used.

### How to use the simulation
The simulation is written in c++14 using the CMake build system. At least version 3.5 of CMake is required. We refer to their
[webiste](https://cmake.org/install "CMake's Homepage") website for instructions on installing CMake, if it is not already present on your system.

The simulation can be build with a terminal using the following steps. It is asusmed that the root of the repository is the working directory.
1. `mkdir build` create a directory which will contain all build files
2. `cd ./build` move to the build directory
3. `cmake ..` generate the build files
4. `make` compile the simulation

Repeat step 4 to recompile. Step 3 needs to be repeated only when new files are added or files are removed. The simulation can be executed by running the `main` executable in `./build/code/`.

Alternatively IDEs could be used that can work with CMAke, for instance CLion or Visual Studio.

### Features
The simulation accepts 5 different command-line arguments which enable and disable different aspects of the simulation. Running the simulation without any arguments breaks execution at every turn allowing the user to request additional information about the turn. The additional information can be printed for every turn automatically by adding `-s` and/or `-r`. Waiting for the user at every turn can be turned off by specifying `-c`.

The last two arguments are used to enable player and debugging mode. In player mode the simulation is turned into a game, where the user controls agent 0. In order for the game to be fair only the actions of all agents are logged, so no information about the other agents' hands is shown. This state is enabled by adding `-p`. All previously mentioned arguments are disabled in player mode.

Debugging mode, run by adding `-d`, also allows the user to control agent 0. The user, however, maintains complete control over what information is shown. This mode is useful to see how the system performs in different states. However, it is not fun to play against the agents in this mode.

```
usage: ./main [-p | (-c -s -r -d)]
Options:
    -d: 		 Enables debugging mode (player mode with logging control enabled)
    -p: 		 Enables player mode, all other options are disabled in player mode.
    -c: 		 Disables control mode, continues through every turn when disabled.
    -s: 		 Enables automatic logging of state information.
    -r: 		 Enables automatic logging of reasoning information.
```