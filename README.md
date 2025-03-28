# Puppet-Hand-Mujoco
construct robot hand used in Mujoco simulation for solving mocap data's hand penetration problem  
>  achieve the function of adjusting robot hand's finger length according to smplx model's beta parameters.
>  connect each joint of robot hand to the respective joint of original data with a tendon spring
>  when joint of original data penetrates an object, the connected robot hand will keep the correct collision with that object
>  use CoACD[1] to convert concave objects to sets of multiple convex objects for collision simulation in Mujoco
