# interbotix_descriptions

## Overview
This package contains the URDFs and meshes for the many X-Series Interbotix arms and turrets. The STL files for each robot are located in a unique folder inside the [meshes](meshes/) directory. Also in the 'meshes' directory is the [interbotix_black.png](meshes/interbotix_black.png) picture. The appearance and texture of the robots come from this picture. Next, the URDFs for the robot are located in the [urdf](urdf/) directory. They are written in 'xacro' format so that users have the ability to customize what parts of the URDF get loaded to the parameter server (see the 'Usage' section below for details). Note that all the other ROS packages in the repo reference this package to launch the robot model.

## Structure
![descriptions_flowchart](images/descriptions_flowchart.png)
This package contains the [description.launch](launch/description.launch) file responsible for loading parts or all of the robot model. It launches up to four nodes as described below:
- **joint_state_publisher** - responsible for parsing the 'robot_description' parameter to find all non-fixed joints and publish a JointState message with those joints defined.
- **joint_state_publisher_gui** - does the same thing as the 'joint_state_publisher' node but with a GUI that allows a user to easily manipulate the joints.
- **robot_state_publisher** - uses the URDF specified by the parameter robot_description and the joint positions from the joint_states topic to calculate the forward kinematics of the robot and publish the results via tf.
- **rviz** - displays the virtual robot model using the transforms in the 'tf' topic.

## Usage
To run this package, type the line below in a terminal. Note that the `robot_name` argument must be specified as the name of one of the URDF files located in the [urdf](/urdf) directory (excluding the '.urdf.xacro' part). For example, to launch the ReactorX 150 arm, type:
```
$ roslaunch interbotix_descriptions description.launch robot_name:=rx150 jnt_pub_gui:=true
```
This is the bare minimum needed to get up and running. Take a look at the table below to see how to further customize with other launch file arguments.

| Argument | Description | Default Value |
| -------- | ----------- | :-----------: |
| robot_name | name of a robot (ex. 'arm1/wx200' or 'wx200') | "" |
| robot_model | only used when launching multiple robots or if `robot_name` contains more than the model type; if that's the case, this should be set to the robot model type (ex. 'wx200'); `robot_name` should then be set to a unique name followed by '$(arg robot_model)' - such as 'arm1/wx200' | '$(arg robot_name)' |
| use_default_gripper_bar | if true, the gripper_bar link is also loaded to the 'robot_description' parameter; if false, the gripper_bar link and any other link past it in the kinematic chain is not loaded to the parameter server. Set to 'false' if you have a custom gripper attachment | true |
| use_default_gripper_fingers | if true, the gripper fingers are also loaded to the 'robot_description' parameter; if false, the gripper fingers and any other link past it in the kinematic chain is not loaded to the parameter server. Set to 'false' if you have custom gripper fingers | true |
| use_world_frame | set this to 'true' if you would like to load a 'world' frame to the 'robot_description' parameter which is located exactly at the 'base_link' frame of the robot; if using multiple robots or if you would like to attach the 'base_link' frame of the robot to a different frame, set this to False | true |  
| external_urdf_loc | the file path to the custom urdf.xacro file that you would like to include in the Interbotix robot's urdf.xacro file| "" |
| load_gazebo_configs | set this to 'true' if Gazebo is being used; it makes sure to also load Gazebo related configs to the 'robot_description' parameter so that the robot models show up black in Gazebo | false |
| jnt_pub_gui | launches the joint_state_publisher GUI | false |
| use_joint_pub | launches the joint_state_publisher node | false |
| use_default_rviz | launches the rviz and static_transform_publisher nodes | true |
| rvizconfig | file path to the config file Rviz should load | refer to [description.launch](launch/description.launch) |
| model | file path to the robot-specific URDF including arguments to be passed in | refer to [description.launch](launch/description.launch) |
