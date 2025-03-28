# Puppet-Hand-Mujoco
construct robot hand used in Mujoco simulation for solving mocap data's hand penetration problem  
>  let robot hand's shape adapt SMPLX model with different beta parameters.
>  ![image](https://github.com/Hongboooooo/Puppet-Hand-Mujoco/blob/main/puppet%20hand.png)  
>  connect each joint of robot hand to the respective joint of original data with a tendon spring
>  ![image](https://github.com/Hongboooooo/Puppet-Hand-Mujoco/blob/main/puppet%20hand%20with%20tendon.png)
>  when joints of original data penetrate an object, the connected robot hand will keep the correct collision with that object  
>  use CoACD to convert concave objects to sets of multiple convex objects for collision simulation in Mujoco  
