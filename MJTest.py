import time
import numpy as np
import mujoco
import mujoco.viewer
import trimesh.transformations as ttf

from extractSMPL_Mesh import extract_finger_tips as eft
from extractSMPL_Mesh import extract_complete_joints_static as ecjs
from xmlGenerator import create_mujoco_xml as cmx

# seq_name = "S01T001"
# seq_name = "S01T018"
# seq_name = "S25T055"
# seq_name = "S35T043"
# seq_name = "S11T021"
# seq_name = "S28T030"
# seq_name = "S30T040"
# seq_name = "S31T088"
# seq_name = "S32T034"
# seq_name = "S32T064"
# seq_name = "S33T029"
# seq_name = "S34T119" # building blocks fine
# seq_name = "S35T030" # washing bowl fine
# seq_name = "S36T091" # camera rotation manipulation good
# seq_name = "S38T047" # washing bowl fine

# seq_name = "S39T101" # building blocks good
# seq_name = "S40T082" # camera manipulation good
# seq_name = "S35T085" # play mouse and laptop good
seq_name = "S36T043" # washing bowl good


object_state_file = f"E:/HIMO_dataset/data/object_pose/{seq_name}.npy"
subject_joints_file = f"E:/export/joints/{seq_name}.npy"

object_pose = np.load(object_state_file, allow_pickle=True).item()
subject_joints = np.load(subject_joints_file, allow_pickle=True)

# switch coordinate for objects' positions
object_transl_switched = {}
for ki in object_pose.keys():
    object_transl_switched[ki] = object_pose[ki]["transl"][:,[2,0,1]]

# switch coordinate for objects' rotations
object_rot_switched = {}
for ki in object_pose.keys():
    object_rot_switched[ki] = object_pose[ki]["rot"][:,[2,0,1]]

# transfer 3x3 rotation matrix to 4 quaternion
object_rot_quaterion = {}
for ki in object_pose.keys():
    white_board = np.zeros([object_rot_switched[ki].shape[0],4])
    for qi in range(object_rot_switched[ki].shape[0]):
        white_board[qi] = ttf.quaternion_from_matrix(object_rot_switched[ki][qi])
    object_rot_quaterion[ki] = white_board



# define joints indices
RIGHT_HAND_JOINTS = [21, 49, 50, 51, 52, 37, 38, 39, 53, 40, 41, 42, 54, 46, 47, 48, 55, 43, 44, 45, 56]
LEFT_HAND_JOINTS = [20, 34, 35, 36, 57, 22, 23, 24, 58, 25, 26, 27, 59, 31, 32, 33, 60, 28, 29, 30, 61]
# RIGHT_HAND_JOINTS_EXPORT = [21, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]
# LEFT_HAND_JOINTS_EXPORT = [20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 57, 58, 59, 60, 61]

# get stand-by joints positions
stand_by_joints_positions = ecjs(seq_name)
stand_by_joints_positions = stand_by_joints_positions[:,[2,0,1]] + np.array([[0,0,1]])

# gather 
right_finger_tips, left_finger_tips = eft(seq_name)
hands_joints = np.concatenate((subject_joints, right_finger_tips, left_finger_tips), axis=1)
# switch coordinate for hands' positions
hands_joints = hands_joints[:,:,[2,0,1]]

# get object name list
object_name_list = []
for ki in object_pose.keys(): 
    object_name_list.append(ki)

# set up Mujoco environment
cmx(object_list = object_name_list, initial_joints_positions = hands_joints[0], standby_joints_positions=stand_by_joints_positions)
model = mujoco.MjModel.from_xml_path("E:/HIMO_dataset/data/MJxml/divided_objects.xml")
data = mujoco.MjData(model)

# 设置模拟参数
sim_time = 0
control_frame_time = 1 / 30  # 30帧每秒
total_frames = object_pose[object_name_list[0]]["rot"].shape[0]
covered_hand_joint_range = 21
covered_hand_node_range = 20
desired_timestep = 0.001  # 1000Hz
model.opt.timestep = desired_timestep

# get object ID
object_body_id_list = {}
for ki in object_pose.keys():
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ki)
    mocap_id = model.body_mocapid[object_body_id]
    object_body_id_list[ki] = mocap_id

# get right hand ID
right_hand_id_list = []
for rhi in range(covered_hand_joint_range):
    rhj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"right_hand_spirit_{rhi}")
    mocap_id = model.body_mocapid[rhj_id] 
    right_hand_id_list.append(mocap_id)

# get left hand ID
left_hand_id_list = []
for lhi in range(covered_hand_joint_range):
    lhj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"left_hand_spirit_{lhi}")
    mocap_id = model.body_mocapid[lhj_id]
    left_hand_id_list.append(mocap_id)

# get motor ID
right_hand_motor_id_list = []
left_hand_motor_id_list = []
for rmi in range(covered_hand_node_range):
    gi = rmi % 4
    gifi = rmi // 4
    rhm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR.value, f"right_{gi}_grade_finger_{gifi}_spring_motor")
    lhm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR.value, f"left_{gi}_grade_finger_{gifi}_spring_motor")
    right_hand_motor_id_list.append(rhm_id)
    left_hand_motor_id_list.append(lhm_id)

# get right hand joints site ID
right_puppet_site_id_list = []
left_puppet_site_id_list = []
for psi in range(covered_hand_node_range):
    gi = psi % 4
    gifi = psi // 4
    right_puppet_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"right_{gi}_grade_finger_{gifi}_site")
    right_puppet_site_id_list.append(right_puppet_site_id)
    left_puppet_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"left_{gi}_grade_finger_{gifi}_site")
    left_puppet_site_id_list.append(left_puppet_site_id)


print("right_hand_id_list:", right_hand_id_list)
print("left_hand_id_list:", left_hand_id_list)

rectified_fingers_joints = np.zeros((total_frames, 62, 3))

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start_time = time.time()
    elapsed_time = 0
    frame = 0
    replay_count = 0
    while viewer.is_running():

        

        # refresh spring control
        data.ctrl[right_hand_motor_id_list] = -0.01
        data.ctrl[left_hand_motor_id_list] = -0.01

        elapsed_time = time.time() - start_time
        # mocap bodies control 
        if elapsed_time >= control_frame_time:
            start_time = time.time()
            frame = frame % total_frames

            # 获取当前帧的姿态
            for oi in object_name_list:
                quat = object_rot_quaterion[oi][frame]  # 四元数
                pos = object_transl_switched[oi][frame]   # 平移向量

                # 设置物体的姿态
                data.mocap_quat[object_body_id_list[oi]] = quat
                data.mocap_pos[object_body_id_list[oi]] = pos

            # get right hand joints positions
            for rhi in range(covered_hand_joint_range):
                pos = hands_joints[frame][RIGHT_HAND_JOINTS[rhi]]
                data.mocap_pos[right_hand_id_list[rhi]] = pos

            # get left hand joints positions
            for lhi in range(covered_hand_joint_range):
                pos = hands_joints[frame][LEFT_HAND_JOINTS[lhi]]
                data.mocap_pos[left_hand_id_list[lhi]] = pos

            if replay_count < 5:
                # print(f"replay_count: {replay_count} | frame: {frame}")
                for pji in range(covered_hand_node_range):
                    rectified_fingers_joints[frame,RIGHT_HAND_JOINTS[pji+1],:] = data.site_xpos[right_puppet_site_id_list[pji]]
                    rectified_fingers_joints[frame,LEFT_HAND_JOINTS[pji+1],:] = data.site_xpos[left_puppet_site_id_list[pji]]
                rectified_fingers_joints[frame,RIGHT_HAND_JOINTS[0],:] = hands_joints[frame][RIGHT_HAND_JOINTS[0]]
                rectified_fingers_joints[frame,LEFT_HAND_JOINTS[0],:] = hands_joints[frame][LEFT_HAND_JOINTS[0]]
            else:
                break # stop simulation after 5 times of looping

                
            frame = frame + 1

            if frame == total_frames:
                np.save(f"E:/export/joints_rectified_by_puppet/{replay_count}/{seq_name}.npy", rectified_fingers_joints)
                replay_count = replay_count + 1
                

        # 推进模拟
        mujoco.mj_step(model, data)

        # 更新查看器
        viewer.sync()


