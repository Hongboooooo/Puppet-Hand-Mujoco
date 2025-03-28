import xml.etree.ElementTree as ET
import os
import numpy as np

def create_mujoco_xml(object_list, initial_joints_positions, standby_joints_positions):

    object_color = "0.75 0.87 0.96 1"
    spirit_joint_color = "1 0.705 0.353 1"
    right_puppet_hand_color = "0.686 1 0.292 1"
    left_puppet_hand_color = "1 0.686 0.292 1"
    

    start_from_frist_frame = False
    jsps = standby_joints_positions
    if start_from_frist_frame == True:
        jsps = initial_joints_positions

    # define joints indices
    RHJ = [21, 49, 50, 51, 52, 37, 38, 39, 53, 40, 41, 42, 54, 46, 47, 48, 55, 43, 44, 45, 56] # right hand joints indices list
    LHJ = [20, 34, 35, 36, 57, 22, 23, 24, 58, 25, 26, 27, 59, 31, 32, 33, 60, 28, 29, 30, 61] # left hand joints indices list

    FNSIL = [] # finger node start indices list
    FNEIL = [] # finger node end indices list

    for fi in range(20):
        finger_count = fi // 4
        finger_node_count = fi % 4
        finger_node_start = finger_node_count + finger_count * 4
        if finger_node_count == 0:
            finger_node_start = 0
        finger_node_end = finger_node_count + 1 + (finger_count * 4)
        FNSIL.append(finger_node_start)
        FNEIL.append(finger_node_end)

    RFL = [] # right finger node length
    RFV = [] # right finger node vector
    LFL = [] # left finger node length
    LFV = [] # left finger node vector
    for fi in range(20):
        RFL.append(np.linalg.norm(jsps[RHJ[FNSIL[fi]]] - jsps[RHJ[FNEIL[fi]]]))
        RFV.append(jsps[RHJ[FNEIL[fi]]] - jsps[RHJ[FNSIL[fi]]])
        LFL.append(np.linalg.norm(jsps[LHJ[FNSIL[fi]]] - jsps[LHJ[FNEIL[fi]]]))
        LFV.append(jsps[LHJ[FNEIL[fi]]] - jsps[LHJ[FNSIL[fi]]])
    
    output_file = "E:/HIMO_dataset/data/MJxml/divided_objects.xml"

    mujoco = ET.Element("mujoco")

    # add assets
    asset = ET.SubElement(mujoco, "asset")
    for obj_name in object_list:
        parts_file = f"E:/HIMO_DATASET/data/object_mesh_divided/{obj_name}"
        for parti in os.listdir(parts_file):
            part_idx, _ = os.path.splitext(parti)
            part_file = f"{parts_file}/{parti}"
            ET.SubElement(asset, "mesh", name=f"{obj_name}_{part_idx}", file=part_file)

    worldbody = ET.SubElement(mujoco, "worldbody")


    # add meshes
    i = 0
    for obj_name in object_list:
        parts_file = f"E:/HIMO_DATASET/data/object_mesh_divided/{obj_name}"
        mocap_body = ET.SubElement(worldbody, "body", name=obj_name, mocap="true", pos=f"{0+i} {0+i} 0")
        for parti in os.listdir(parts_file):
            part_idx, _ = os.path.splitext(parti)
            ET.SubElement(mocap_body, "geom", type="mesh", mesh=f"{obj_name}_{part_idx}", rgba=object_color, pos=f"0 0 0")
        i = i + 1

    # add right hand spirit joints
    right_hand_joint_spirits = []
    for ji in range(21):
        rh_joint_posi = f"{jsps[RHJ[ji]][0]} {jsps[RHJ[ji]][1]} {jsps[RHJ[ji]][2]}"
        right_hand_joint_spirits.append(ET.SubElement(worldbody, "body", name=f"right_hand_spirit_{ji}", mocap="true", pos=rh_joint_posi))
        ET.SubElement(right_hand_joint_spirits[ji], 
                      "geom", 
                      name=f"right_hand_spirit_{ji}_geom", 
                      type="sphere", 
                      size="0.006", 
                      rgba=f"{ji/20} {ji/20} {ji/20} 1", 
                      pos="0 0 0", 
                      contype="0", 
                      conaffinity="0")
        ET.SubElement(right_hand_joint_spirits[ji], "site", name=f"right_spirit_joint_{ji}_site", pos="0 0 0")
        
    # add left hand spirit joints
    left_hand_joint_spirits = []
    for ji in range(21):
        lh_joint_posi = f"{jsps[LHJ[ji]][0]} {jsps[LHJ[ji]][1]} {jsps[LHJ[ji]][2]}"
        left_hand_joint_spirits.append(ET.SubElement(worldbody, "body", name=f"left_hand_spirit_{ji}", mocap="true", pos=lh_joint_posi))
        ET.SubElement(left_hand_joint_spirits[ji], 
                      "geom", 
                      name=f"left_hand_spirit_{ji}_geom", 
                      type="sphere", 
                      size="0.006", 
                      rgba=f"{ji/20} {ji/20} {ji/20} 1", 
                      pos="0 0 0", 
                      contype="0", 
                      conaffinity="0")
        ET.SubElement(left_hand_joint_spirits[ji], "site", name=f"left_spirit_joint_{ji}_site", pos="0 0 0")
        
    right_hand_thumb_axis = np.cross(RFV[0],np.array([0,-1,0]))
    left_hand_thumb_axis = np.cross(LFV[0],np.array([0,1,0]))
    right_hand_thumb_axis_addition = np.cross(RFV[0], right_hand_thumb_axis)
    left_hand_thumb_axis_addition = np.cross(LFV[0], left_hand_thumb_axis)
        
    # add right puppet hand finger node grade: root
    right_finger_node_list = []

    right_palm = ET.SubElement(right_hand_joint_spirits[0], "body", name=f"right_palm")
    ET.SubElement(right_palm, "joint", name=f"right_palm_joint", type="ball", damping="0.001")
    for rfi in range(5):
        connect_idx = 4 * rfi
        
        ET.SubElement(right_palm,
                        "geom",
                        name=f"right_{0}_grade_finger_{rfi}_collision",
                        type="capsule",
                        size=f"0.006 {RFL[connect_idx]}",
                        rgba=right_puppet_hand_color,
                        fromto=f"0 0 0 {RFV[connect_idx][0]} {RFV[connect_idx][1]} {RFV[connect_idx][2]}")
        
        ET.SubElement(right_palm, 
                        "site", 
                        name=f"right_{0}_grade_finger_{rfi}_site", 
                        pos=f"{RFV[connect_idx][0]} {RFV[connect_idx][1]} {RFV[connect_idx][2]}")
        
        

        tendon = ET.SubElement(mujoco, "tendon")
        spring = ET.SubElement(tendon, "spatial", name=f"right_{0}_grade_finger_{rfi}_spring", width="0.002", rgba=spirit_joint_color, damping="1", springlength="0")
        ET.SubElement(spring, "site", site=f"right_{0}_grade_finger_{rfi}_site")
        ET.SubElement(spring, "site", site=f"right_spirit_joint_{connect_idx+1}_site")

        actuator = ET.SubElement(mujoco, "actuator")
        ET.SubElement(actuator, 
                        "motor", 
                        name=f"right_{0}_grade_finger_{rfi}_spring_motor", 
                        tendon=f"right_{0}_grade_finger_{rfi}_spring", 
                        gear="20", 
                        ctrlrange="-0.01 0.01")

    right_finger_node_list.append(right_palm)

    # add right puppet hand finger node grade: 1st
    for gifi in range(5):
        gi = 1
        connect_idx = 4 * gifi + 1
        previous_fi = 0
        if gifi == 0:
            hinger_set_1st = ET.SubElement(right_finger_node_list[previous_fi], 
                                                "body", 
                                                name=f"right_{gi}_grade_finger_{gifi}", 
                                                pos=f"{RFV[connect_idx-1][0]} {RFV[connect_idx-1][1]} {RFV[connect_idx-1][2]}")
            ET.SubElement(hinger_set_1st, 
                            "joint", 
                            name=f"right_{gi}_grade_finger_{gifi}_joint_y", 
                            type="hinge", 
                            axis=f"0 -1 0", 
                            pos="0 0 0", 
                            damping="0.001",
                            limited="true",
                            range="-60 45")
            ET.SubElement(hinger_set_1st,
                            "geom",
                            name=f"right_{gi}_grade_finger_{gifi}_set_y",
                            type="sphere",
                            size=f"0.001",
                            rgba="0.8 0.8 0.8 1",
                            pos=f"0 0 0"
                            )
            
            hinger_set_2nd = ET.SubElement(hinger_set_1st, 
                                        "body", 
                                        name=f"right_{gi}_grade_finger_{gifi}_set", 
                                        pos=f"0 0 0")
            ET.SubElement(hinger_set_2nd, 
                            "joint", 
                            name=f"right_{gi}_grade_finger_{gifi}_joint_x", 
                            type="hinge", 
                            axis=f"{right_hand_thumb_axis_addition[0]} {right_hand_thumb_axis_addition[1]} {right_hand_thumb_axis_addition[2]}", 
                            pos="0 0 0", 
                            damping="0.001",
                            limited="true",
                            range="-15 15")
            ET.SubElement(hinger_set_2nd,
                            "geom",
                            name=f"right_{gi}_grade_finger_{gifi}_set",
                            type="sphere",
                            size=f"0.0005",
                            rgba="0.6 0.6 0.6 1",
                            pos=f"0 0 0"
                            )

            right_finger_node = ET.SubElement(hinger_set_2nd, 
                                        "body", 
                                        name=f"right_{gi}_grade_finger_{gifi}_finger", 
                                        pos=f"0 0 0")
            ET.SubElement(right_finger_node, 
                            "joint", 
                            name=f"right_{gi}_grade_finger_{gifi}_joint_z", 
                            type="hinge", 
                            axis=f"{right_hand_thumb_axis[0]} {right_hand_thumb_axis[1]} {right_hand_thumb_axis[2]}", 
                            pos="0 0 0", 
                            damping="0.001", 
                            limited="true", 
                            range="-30 75")
            ET.SubElement(right_finger_node, 
                            "site", 
                            name=f"right_{gi}_grade_finger_{gifi}_site", 
                            pos=f"{RFV[connect_idx][0]} {RFV[connect_idx][1]} {RFV[connect_idx][2]}")
            ET.SubElement(right_finger_node,
                            "geom",
                            name=f"right_{gi}_grade_finger_{gifi}_collision",
                            type="capsule",
                            size=f"0.006 {RFL[connect_idx]}",
                            rgba=right_puppet_hand_color,
                            fromto=f"0 0 0 {RFV[connect_idx][0]} {RFV[connect_idx][1]} {RFV[connect_idx][2]}")
            right_finger_node_list.append(right_finger_node)

            tendon = ET.SubElement(mujoco, "tendon")
            spring = ET.SubElement(tendon, "spatial", 
                                    name=f"right_{gi}_grade_finger_{gifi}_spring", 
                                    width="0.002", 
                                    rgba=spirit_joint_color, 
                                    damping="1", 
                                    springlength="0")
            ET.SubElement(spring, "site", site=f"right_{gi}_grade_finger_{gifi}_site")
            ET.SubElement(spring, "site", site=f"right_spirit_joint_{connect_idx+1}_site")

            actuator = ET.SubElement(mujoco, "actuator")
            ET.SubElement(actuator, "motor", 
                            name=f"right_{gi}_grade_finger_{gifi}_spring_motor", 
                            tendon=f"right_{gi}_grade_finger_{gifi}_spring", 
                            gear="10", 
                            ctrlrange="-0.01 0.01")
        else:
            hinger_set = ET.SubElement(right_finger_node_list[previous_fi], 
                                                "body", 
                                                name=f"right_{gi}_grade_finger_{gifi}", 
                                                pos=f"{RFV[connect_idx-1][0]} {RFV[connect_idx-1][1]} {RFV[connect_idx-1][2]}")
            ET.SubElement(hinger_set, 
                            "joint", 
                            name=f"right_{gi}_grade_finger_{gifi}_joint_z", 
                            type="hinge", 
                            axis=f"0 0 1", 
                            pos="0 0 0", 
                            damping="0.001",
                            limited="true",
                            range="-30 30")
            ET.SubElement(hinger_set,
                            "geom",
                            name=f"right_{gi}_grade_finger_{gifi}_set",
                            type="sphere",
                            size=f"0.001",
                            rgba="0.8 0.8 0.8 1",
                            pos=f"0 0 0"
                            )

            right_finger_node = ET.SubElement(hinger_set, 
                                        "body", 
                                        name=f"right_{gi}_grade_finger_{gifi}_set", 
                                        pos=f"0 0 0")
            ET.SubElement(right_finger_node, 
                            "joint", 
                            name=f"right_{gi}_grade_finger_{gifi}_joint_x", type="hinge", 
                            axis=f"{-RFV[connect_idx][1]} {RFV[connect_idx][0]} 0", 
                            pos="0 0 0", 
                            damping="0.001", 
                            limited="true", 
                            range="-30 90")
            ET.SubElement(right_finger_node, 
                            "site", 
                            name=f"right_{gi}_grade_finger_{gifi}_site", 
                            pos=f"{RFV[connect_idx][0]} {RFV[connect_idx][1]} {RFV[connect_idx][2]}")
            ET.SubElement(right_finger_node,
                            "geom",
                            name=f"right_{gi}_grade_finger_{gifi}_collision",
                            type="capsule",
                            size=f"0.006 {RFL[connect_idx]}",
                            rgba=right_puppet_hand_color,
                            fromto=f"0 0 0 {RFV[connect_idx][0]} {RFV[connect_idx][1]} {RFV[connect_idx][2]}")
            right_finger_node_list.append(right_finger_node)

            tendon = ET.SubElement(mujoco, "tendon")
            spring = ET.SubElement(tendon, "spatial", 
                                    name=f"right_{gi}_grade_finger_{gifi}_spring", 
                                    width="0.002", 
                                    rgba=spirit_joint_color, 
                                    damping="1", 
                                    springlength="0")
            ET.SubElement(spring, "site", site=f"right_{gi}_grade_finger_{gifi}_site")
            ET.SubElement(spring, "site", site=f"right_spirit_joint_{connect_idx+1}_site")

            actuator = ET.SubElement(mujoco, "actuator")
            ET.SubElement(actuator, "motor", 
                            name=f"right_{gi}_grade_finger_{gifi}_spring_motor", 
                            tendon=f"right_{gi}_grade_finger_{gifi}_spring", 
                            gear="10", 
                            ctrlrange="-0.01 0.01")

    # add right puppet hand finger node grade: 2nd & 3rd
    for gi in range(2,4):
        for gifi in range(5):
            if gifi == 0:
                connect_idx = 4 * gifi + gi
                previous_fi = 5 * (gi - 1) + gifi - 4
                right_finger_node = ET.SubElement(right_finger_node_list[previous_fi], 
                                                    "body", 
                                                    name=f"right_{gi}_grade_finger_{gifi}", 
                                                    pos=f"{RFV[connect_idx-1][0]} {RFV[connect_idx-1][1]} {RFV[connect_idx-1][2]}")
                ET.SubElement(right_finger_node, 
                                "joint", 
                                name=f"right_{gi}_grade_finger_{gifi}_joint", 
                                type="hinge", 
                                axis=f"{right_hand_thumb_axis[0]} {right_hand_thumb_axis[1]} {right_hand_thumb_axis[2]}", 
                                pos="0 0 0", 
                                damping="0.001",
                                limited="true",
                                range="-5 90")
                ET.SubElement(right_finger_node, 
                                "site", 
                                name=f"right_{gi}_grade_finger_{gifi}_site", 
                                pos=f"{RFV[connect_idx][0]} {RFV[connect_idx][1]} {RFV[connect_idx][2]}")
                ET.SubElement(right_finger_node,
                                "geom",
                                name=f"right_{gi}_grade_finger_{gifi}_collision",
                                type="capsule",
                                size=f"0.006 {RFL[connect_idx]}",
                                rgba=right_puppet_hand_color,
                                fromto=f"0 0 0 {RFV[connect_idx][0]} {RFV[connect_idx][1]} {RFV[connect_idx][2]}")
                right_finger_node_list.append(right_finger_node)

                tendon = ET.SubElement(mujoco, "tendon")
                spring = ET.SubElement(tendon, "spatial", 
                                        name=f"right_{gi}_grade_finger_{gifi}_spring", 
                                        width="0.002", 
                                        rgba=spirit_joint_color, 
                                        damping="1", 
                                        springlength="0")
                ET.SubElement(spring, "site", site=f"right_{gi}_grade_finger_{gifi}_site")
                ET.SubElement(spring, "site", site=f"right_spirit_joint_{connect_idx+1}_site")

                actuator = ET.SubElement(mujoco, "actuator")
                ET.SubElement(actuator, "motor", 
                                name=f"right_{gi}_grade_finger_{gifi}_spring_motor", 
                                tendon=f"right_{gi}_grade_finger_{gifi}_spring", 
                                gear="10", 
                                ctrlrange="-0.01 0.01")
            else:
                connect_idx = 4 * gifi + gi
                previous_fi = 5 * (gi - 1) + gifi - 4
                right_finger_node = ET.SubElement(right_finger_node_list[previous_fi], 
                                                    "body", 
                                                    name=f"right_{gi}_grade_finger_{gifi}", 
                                                    pos=f"{RFV[connect_idx-1][0]} {RFV[connect_idx-1][1]} {RFV[connect_idx-1][2]}")
                ET.SubElement(right_finger_node, 
                                "joint", 
                                name=f"right_{gi}_grade_finger_{gifi}_joint", 
                                type="hinge", 
                                axis=f"{-RFV[connect_idx][1]} {RFV[connect_idx][0]} 0", 
                                pos="0 0 0", 
                                damping="0.001",
                                limited="true",
                                range="-5 90")
                ET.SubElement(right_finger_node, 
                                "site", 
                                name=f"right_{gi}_grade_finger_{gifi}_site", 
                                pos=f"{RFV[connect_idx][0]} {RFV[connect_idx][1]} {RFV[connect_idx][2]}")
                ET.SubElement(right_finger_node,
                                "geom",
                                name=f"right_{gi}_grade_finger_{gifi}_collision",
                                type="capsule",
                                size=f"0.006 {RFL[connect_idx]}",
                                rgba=right_puppet_hand_color,
                                fromto=f"0 0 0 {RFV[connect_idx][0]} {RFV[connect_idx][1]} {RFV[connect_idx][2]}")
                right_finger_node_list.append(right_finger_node)

                tendon = ET.SubElement(mujoco, "tendon")
                spring = ET.SubElement(tendon, "spatial", 
                                        name=f"right_{gi}_grade_finger_{gifi}_spring", 
                                        width="0.002", 
                                        rgba=spirit_joint_color, 
                                        damping="1", 
                                        springlength="0")
                ET.SubElement(spring, "site", site=f"right_{gi}_grade_finger_{gifi}_site")
                ET.SubElement(spring, "site", site=f"right_spirit_joint_{connect_idx+1}_site")

                actuator = ET.SubElement(mujoco, "actuator")
                ET.SubElement(actuator, "motor", 
                                name=f"right_{gi}_grade_finger_{gifi}_spring_motor", 
                                tendon=f"right_{gi}_grade_finger_{gifi}_spring", 
                                gear="10", 
                                ctrlrange="-0.01 0.01")
                

    # add left puppet hand finger node grade: root
    left_finger_node_list = []

    left_palm = ET.SubElement(left_hand_joint_spirits[0], "body", name=f"left_palm")
    ET.SubElement(left_palm, "joint", name=f"left_palm_joint", type="ball", damping="0.001")
    for rfi in range(5):
        connect_idx = 4 * rfi
        
        ET.SubElement(left_palm,
                        "geom",
                        name=f"left_{0}_grade_finger_{rfi}_collision",
                        type="capsule",
                        size=f"0.006 {LFL[connect_idx]}",
                        rgba=left_puppet_hand_color,
                        fromto=f"0 0 0 {LFV[connect_idx][0]} {LFV[connect_idx][1]} {LFV[connect_idx][2]}")
        
        ET.SubElement(left_palm, 
                        "site", 
                        name=f"left_{0}_grade_finger_{rfi}_site", 
                        pos=f"{LFV[connect_idx][0]} {LFV[connect_idx][1]} {LFV[connect_idx][2]}")
        
        

        tendon = ET.SubElement(mujoco, "tendon")
        spring = ET.SubElement(tendon, "spatial", name=f"left_{0}_grade_finger_{rfi}_spring", width="0.002", rgba=spirit_joint_color, damping="1", springlength="0")
        ET.SubElement(spring, "site", site=f"left_{0}_grade_finger_{rfi}_site")
        ET.SubElement(spring, "site", site=f"left_spirit_joint_{connect_idx+1}_site")

        actuator = ET.SubElement(mujoco, "actuator")
        ET.SubElement(actuator, 
                        "motor", 
                        name=f"left_{0}_grade_finger_{rfi}_spring_motor", 
                        tendon=f"left_{0}_grade_finger_{rfi}_spring", 
                        gear="20", 
                        ctrlrange="-0.01 0.01")

    left_finger_node_list.append(left_palm)

    # add left puppet hand finger node grade: 1st
    for gifi in range(5):
        gi = 1
        connect_idx = 4 * gifi + 1
        previous_fi = 0
        if gifi == 0:
            hinger_set_1st = ET.SubElement(left_finger_node_list[previous_fi], 
                                                "body", 
                                                name=f"left_{gi}_grade_finger_{gifi}", 
                                                pos=f"{LFV[connect_idx-1][0]} {LFV[connect_idx-1][1]} {LFV[connect_idx-1][2]}")
            ET.SubElement(hinger_set_1st, 
                            "joint", 
                            name=f"left_{gi}_grade_finger_{gifi}_joint_y", 
                            type="hinge", 
                            axis=f"0 1 0", 
                            pos="0 0 0", 
                            damping="0.001",
                            limited="true",
                            range="-60 45")
            ET.SubElement(hinger_set_1st,
                            "geom",
                            name=f"left_{gi}_grade_finger_{gifi}_set_y",
                            type="sphere",
                            size=f"0.001",
                            rgba="0.8 0.8 0.8 1",
                            pos=f"0 0 0"
                            )
            
            hinger_set_2nd = ET.SubElement(hinger_set_1st, 
                                        "body", 
                                        name=f"left_{gi}_grade_finger_{gifi}_set", 
                                        pos=f"0 0 0")
            ET.SubElement(hinger_set_2nd, 
                            "joint", 
                            name=f"left_{gi}_grade_finger_{gifi}_joint_x", 
                            type="hinge", 
                            axis=f"{left_hand_thumb_axis_addition[0]} {left_hand_thumb_axis_addition[1]} {left_hand_thumb_axis_addition[2]}", 
                            pos="0 0 0", 
                            damping="0.001",
                            limited="true",
                            range="-15 15")
            ET.SubElement(hinger_set_2nd,
                            "geom",
                            name=f"left_{gi}_grade_finger_{gifi}_set",
                            type="sphere",
                            size=f"0.0005",
                            rgba="0.6 0.6 0.6 1",
                            pos=f"0 0 0"
                            )

            left_finger_node = ET.SubElement(hinger_set_2nd, 
                                        "body", 
                                        name=f"left_{gi}_grade_finger_{gifi}_finger", 
                                        pos=f"0 0 0")
            ET.SubElement(left_finger_node, 
                            "joint", 
                            name=f"left_{gi}_grade_finger_{gifi}_joint_z", 
                            type="hinge", 
                            axis=f"{left_hand_thumb_axis[0]} {left_hand_thumb_axis[1]} {left_hand_thumb_axis[2]}", 
                            pos="0 0 0", 
                            damping="0.001", 
                            limited="true", 
                            range="-30 75")
            ET.SubElement(left_finger_node, 
                            "site", 
                            name=f"left_{gi}_grade_finger_{gifi}_site", 
                            pos=f"{LFV[connect_idx][0]} {LFV[connect_idx][1]} {LFV[connect_idx][2]}")
            ET.SubElement(left_finger_node,
                            "geom",
                            name=f"left_{gi}_grade_finger_{gifi}_collision",
                            type="capsule",
                            size=f"0.006 {LFL[connect_idx]}",
                            rgba=left_puppet_hand_color,
                            fromto=f"0 0 0 {LFV[connect_idx][0]} {LFV[connect_idx][1]} {LFV[connect_idx][2]}")
            left_finger_node_list.append(left_finger_node)

            tendon = ET.SubElement(mujoco, "tendon")
            spring = ET.SubElement(tendon, "spatial", 
                                    name=f"left_{gi}_grade_finger_{gifi}_spring", 
                                    width="0.002", 
                                    rgba=spirit_joint_color, 
                                    damping="1", 
                                    springlength="0")
            ET.SubElement(spring, "site", site=f"left_{gi}_grade_finger_{gifi}_site")
            ET.SubElement(spring, "site", site=f"left_spirit_joint_{connect_idx+1}_site")

            actuator = ET.SubElement(mujoco, "actuator")
            ET.SubElement(actuator, "motor", 
                            name=f"left_{gi}_grade_finger_{gifi}_spring_motor", 
                            tendon=f"left_{gi}_grade_finger_{gifi}_spring", 
                            gear="10", 
                            ctrlrange="-0.01 0.01")
        else:
            hinger_set = ET.SubElement(left_finger_node_list[previous_fi], 
                                                "body", 
                                                name=f"left_{gi}_grade_finger_{gifi}", 
                                                pos=f"{LFV[connect_idx-1][0]} {LFV[connect_idx-1][1]} {LFV[connect_idx-1][2]}")
            ET.SubElement(hinger_set, 
                            "joint", 
                            name=f"left_{gi}_grade_finger_{gifi}_joint_z", 
                            type="hinge", 
                            axis=f"0 0 1", 
                            pos="0 0 0", 
                            damping="0.001",
                            limited="true",
                            range="-30 30")
            ET.SubElement(hinger_set,
                            "geom",
                            name=f"left_{gi}_grade_finger_{gifi}_set",
                            type="sphere",
                            size=f"0.001",
                            rgba="0.8 0.8 0.8 1",
                            pos=f"0 0 0"
                            )

            left_finger_node = ET.SubElement(hinger_set, 
                                        "body", 
                                        name=f"left_{gi}_grade_finger_{gifi}_set", 
                                        pos=f"0 0 0")
            ET.SubElement(left_finger_node, 
                            "joint", 
                            name=f"left_{gi}_grade_finger_{gifi}_joint_x", type="hinge", 
                            axis=f"{-LFV[connect_idx][1]} {LFV[connect_idx][0]} 0", 
                            pos="0 0 0", 
                            damping="0.001", 
                            limited="true", 
                            range="-30 90")
            ET.SubElement(left_finger_node, 
                            "site", 
                            name=f"left_{gi}_grade_finger_{gifi}_site", 
                            pos=f"{LFV[connect_idx][0]} {LFV[connect_idx][1]} {LFV[connect_idx][2]}")
            ET.SubElement(left_finger_node,
                            "geom",
                            name=f"left_{gi}_grade_finger_{gifi}_collision",
                            type="capsule",
                            size=f"0.006 {LFL[connect_idx]}",
                            rgba=left_puppet_hand_color,
                            fromto=f"0 0 0 {LFV[connect_idx][0]} {LFV[connect_idx][1]} {LFV[connect_idx][2]}")
            left_finger_node_list.append(left_finger_node)

            tendon = ET.SubElement(mujoco, "tendon")
            spring = ET.SubElement(tendon, "spatial", 
                                    name=f"left_{gi}_grade_finger_{gifi}_spring", 
                                    width="0.002", 
                                    rgba=spirit_joint_color, 
                                    damping="1", 
                                    springlength="0")
            ET.SubElement(spring, "site", site=f"left_{gi}_grade_finger_{gifi}_site")
            ET.SubElement(spring, "site", site=f"left_spirit_joint_{connect_idx+1}_site")

            actuator = ET.SubElement(mujoco, "actuator")
            ET.SubElement(actuator, "motor", 
                            name=f"left_{gi}_grade_finger_{gifi}_spring_motor", 
                            tendon=f"left_{gi}_grade_finger_{gifi}_spring", 
                            gear="10", 
                            ctrlrange="-0.01 0.01")

    # add left puppet hand finger node grade: 2nd & 3rd
    for gi in range(2,4):
        for gifi in range(5):
            if gifi == 0:
                connect_idx = 4 * gifi + gi
                previous_fi = 5 * (gi - 1) + gifi - 4
                left_finger_node = ET.SubElement(left_finger_node_list[previous_fi], 
                                                    "body", 
                                                    name=f"left_{gi}_grade_finger_{gifi}", 
                                                    pos=f"{LFV[connect_idx-1][0]} {LFV[connect_idx-1][1]} {LFV[connect_idx-1][2]}")
                ET.SubElement(left_finger_node, 
                                "joint", 
                                name=f"left_{gi}_grade_finger_{gifi}_joint", 
                                type="hinge", 
                                axis=f"{left_hand_thumb_axis[0]} {left_hand_thumb_axis[1]} {left_hand_thumb_axis[2]}", 
                                pos="0 0 0", 
                                damping="0.001",
                                limited="true",
                                range="-5 90")
                ET.SubElement(left_finger_node, 
                                "site", 
                                name=f"left_{gi}_grade_finger_{gifi}_site", 
                                pos=f"{LFV[connect_idx][0]} {LFV[connect_idx][1]} {LFV[connect_idx][2]}")
                ET.SubElement(left_finger_node,
                                "geom",
                                name=f"left_{gi}_grade_finger_{gifi}_collision",
                                type="capsule",
                                size=f"0.006 {LFL[connect_idx]}",
                                rgba=left_puppet_hand_color,
                                fromto=f"0 0 0 {LFV[connect_idx][0]} {LFV[connect_idx][1]} {LFV[connect_idx][2]}")
                left_finger_node_list.append(left_finger_node)

                tendon = ET.SubElement(mujoco, "tendon")
                spring = ET.SubElement(tendon, "spatial", 
                                        name=f"left_{gi}_grade_finger_{gifi}_spring", 
                                        width="0.002", 
                                        rgba=spirit_joint_color, 
                                        damping="1", 
                                        springlength="0")
                ET.SubElement(spring, "site", site=f"left_{gi}_grade_finger_{gifi}_site")
                ET.SubElement(spring, "site", site=f"left_spirit_joint_{connect_idx+1}_site")

                actuator = ET.SubElement(mujoco, "actuator")
                ET.SubElement(actuator, "motor", 
                                name=f"left_{gi}_grade_finger_{gifi}_spring_motor", 
                                tendon=f"left_{gi}_grade_finger_{gifi}_spring", 
                                gear="10", 
                                ctrlrange="-0.01 0.01")
            else:
                connect_idx = 4 * gifi + gi
                previous_fi = 5 * (gi - 1) + gifi - 4
                left_finger_node = ET.SubElement(left_finger_node_list[previous_fi], 
                                                    "body", 
                                                    name=f"left_{gi}_grade_finger_{gifi}", 
                                                    pos=f"{LFV[connect_idx-1][0]} {LFV[connect_idx-1][1]} {LFV[connect_idx-1][2]}")
                ET.SubElement(left_finger_node, 
                                "joint", 
                                name=f"left_{gi}_grade_finger_{gifi}_joint", 
                                type="hinge", 
                                axis=f"{-LFV[connect_idx][1]} {LFV[connect_idx][0]} 0", 
                                pos="0 0 0", 
                                damping="0.001",
                                limited="true",
                                range="-5 90")
                ET.SubElement(left_finger_node, 
                                "site", 
                                name=f"left_{gi}_grade_finger_{gifi}_site", 
                                pos=f"{LFV[connect_idx][0]} {LFV[connect_idx][1]} {LFV[connect_idx][2]}")
                ET.SubElement(left_finger_node,
                                "geom",
                                name=f"left_{gi}_grade_finger_{gifi}_collision",
                                type="capsule",
                                size=f"0.006 {LFL[connect_idx]}",
                                rgba=left_puppet_hand_color,
                                fromto=f"0 0 0 {LFV[connect_idx][0]} {LFV[connect_idx][1]} {LFV[connect_idx][2]}")
                left_finger_node_list.append(left_finger_node)

                tendon = ET.SubElement(mujoco, "tendon")
                spring = ET.SubElement(tendon, "spatial", 
                                        name=f"left_{gi}_grade_finger_{gifi}_spring", 
                                        width="0.002", 
                                        rgba=spirit_joint_color, 
                                        damping="1", 
                                        springlength="0")
                ET.SubElement(spring, "site", site=f"left_{gi}_grade_finger_{gifi}_site")
                ET.SubElement(spring, "site", site=f"left_spirit_joint_{connect_idx+1}_site")

                actuator = ET.SubElement(mujoco, "actuator")
                ET.SubElement(actuator, "motor", 
                                name=f"left_{gi}_grade_finger_{gifi}_spring_motor", 
                                tendon=f"left_{gi}_grade_finger_{gifi}_spring", 
                                gear="10", 
                                ctrlrange="-0.01 0.01")



    # add coordinate
    color_mask = [[1,0,0],[0,1,0],[0,0,1]]
    no_collision_body = ET.SubElement(worldbody, "body", name=f"no_collision_sphere_body")
    for ci in range(len(color_mask)):
        ET.SubElement(no_collision_body, 
                      "geom", 
                      name=f"ghost_capsule_{ci}", 
                      type="capsule", 
                      size="0.005 1", 
                      rgba=f"{color_mask[ci][0]} {color_mask[ci][1]} {color_mask[ci][2]} 1", 
                      fromto=f"0 0 0 {color_mask[ci][0]} {color_mask[ci][1]} {color_mask[ci][2]}", 
                      contype="0", 
                      conaffinity="0")

        
    # add ground
    ET.SubElement(worldbody, "geom", name=f"ground", type="plane", size="10 10 0.1", pos="0 0 0", rgba="0.9 0.9 0.9 1")


    tree = ET.ElementTree(mujoco)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


    print(f"MuJoCo model saved to {output_file}")