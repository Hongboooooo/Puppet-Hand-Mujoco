# Puppet-Hand-Mujoco  
Solve hand penetration problem frequently seen in mocap datasets by using physical simulation  
>1. construct rigid-body hand according to Mano's beta parameters in Mujoco simulation  
>2. connect each joint of robot hand to the respective joint of original data with a tendon spring
>  ![image](https://github.com/Hongboooooo/Puppet-Hand-Mujoco/blob/main/puppet%20hand%20with%20tendon.png)  

Correction procedure can be seen in the video below  
>  1st segment shows original mocap data. 2nd segment applies puppet hand in physics simulation; 3rd segment shows rectified data  
![image](https://github.com/Hongboooooo/Puppet-Hand-Mujoco/blob/main/S40T082front.gif)
