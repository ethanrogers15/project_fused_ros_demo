Project FUSED ROS Demo Environment README

Fusion-Based Utilization and Synthesis of Efficient Detections

Check out our website here for background information about the project: 
https://sites.google.com/view/projectfused

Point of Contact: Ethan Rogers, ethan.c.rogers@gmail.com

This repository contains a demo derived from the original ROS environment
used for calibration, data recording, and fusion workflow deployment.

The main FUSED environment is located here:
https://github.com/ethanrogers15/project_fused

The ROS2 environment repository is located here: 
https://github.com/ethanrogers15/project_fused_ros

The model training environment is located here:
https://github.com/ethanrogers15/mediapipe_model_maker

To run this virtual environment, you will need Docker Desktop and VSCode with
the Remote Development & Dev Containers extensions. The virtual environments
for Project FUSED were built on a Windows laptop, but we used WSL with Ubuntu
to interface with the ROS environment. So, to run this demo, you will need
WSL and a version of Ubuntu set up if you are using a Windows machine. 

You will need to clone the repository into the WSL file system. After opening 
the directory in VSCode, you will be prompted to "build" the environment as a 
Dev Container. After starting the build, it may take a long time to complete.

Once the environment is built, the next step is to build the ROS environment
inside of the Dev Container. Type in the following commands in the terminal:  


source /opt/ros/humble/setup.bash  
rosdep update  
rosdep install --from-paths src --ignore-src -r -y  
colcon build  
source install/setup.bash  


If any changes are made inside the 'src' directory, you should re-enter the 
following commands to update:


colcon build  
source install/setup.bash  


After the camera_combiner package has been built, make sure that the 
environment's Python interpreter is in use when selecting a Python file, and 
you should be good to go!

Note that in order to run the demo, you will need the 'images' directory added
under the root directory 'project_fused_ros_demo'. The images folder is large, 
so it could not be added to Github easily. Here is a Google Drive link for the
data; copy the folder into the environment.

https://drive.google.com/drive/folders/1KNZYa9dd96nyBZ3kJqvgcqFy7q6X14Fn?usp=drive_link

Before running the demo, you will need a way to view RVIZ2 to see the detections.
We used the following GUI viewer, and you will either need to install and run this
before starting the demo or some similar GUI viewer:

https://sourceforge.net/projects/vcxsrv/

To run the demo, type the following into the terminal:

ros2 launch camera_combiner workflow.launch.py

You should be able to see a window with the original object detections from the individual
sensors, and there should be a separate window showing the object detections resulting from
the fusion algorithm. Feel free to change the launch file directory input to different
ROS bags so you can view different scenarios. Again, you will need the images folder
from the Google Drive link uploaded to the environment.
