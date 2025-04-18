# Puppet-Hand-Mujoco
construct rigid-body hand used in Mujoco simulation for solving hand penetration problem frequently seen in mocap datasets 
>  connect each joint of robot hand to the respective joint of original data with a tendon spring
>  ![image](https://github.com/Hongboooooo/Puppet-Hand-Mujoco/blob/main/puppet%20hand%20with%20tendon.png)  
>  when joints of original data penetrate an object, the connected robot hand will keep the correct collision with that object  

Correction procedure can be seen in the video belong  
>  1st segment shows original mocap data. 2nd segment applies puppet hand in physics simulation; 3rd segment shows rectified data  
![image](https://github.com/Hongboooooo/Puppet-Hand-Mujoco/blob/main/S40T082front.gif)
