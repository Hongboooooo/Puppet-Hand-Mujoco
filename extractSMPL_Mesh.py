import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
import torch
import os
import random
import h5py
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.configuration import CONFIG as C
from aitviewer.utils import to_torch

dtype = C.f_precision
device = C.device

def extract_complete_joints_static(label):
    subject_smplx_file = f"E:/HIMO_dataset/data/smplx/{label}.npz"
    subject_smplx = np.load(subject_smplx_file, allow_pickle=True)

    poses_root = np.zeros((1,3))
    poses_body = np.zeros((1,21*3))
    poses_right_hand = np.zeros((1,15*3))
    poses_left_hand = np.zeros((1,15*3))
    trans = np.zeros((1,3))
    betas = subject_smplx["betas"]

    poses_root_torch = to_torch(poses_root, dtype=dtype, device=device)
    poses_body_torch = to_torch(poses_body, dtype=dtype, device=device)
    poses_right_hand_torch = to_torch(poses_right_hand, dtype=dtype, device=device)
    poses_left_hand_torch = to_torch(poses_left_hand, dtype=dtype, device=device)
    trans_torch = to_torch(trans, dtype=dtype, device=device)
    betas_torch = to_torch(betas, dtype=dtype, device=device)

    smplx_layer = SMPLLayer(model_type='smplx',gender='neutral',num_betas=10,device=C.device)
    _, joints = smplx_layer(
                            poses_root=poses_root_torch,
                            poses_body=poses_body_torch,
                            poses_right_hand=poses_right_hand_torch,
                            poses_left_hand=poses_left_hand_torch,
                            trans=trans_torch,
                            betas=betas_torch,
                        )

    selected_joints = joints[:,np.concatenate([range(0,22),range(25,55)]),:].cpu().numpy()

    smplx_seq = SMPLSequence(
                            smpl_layer=smplx_layer,
                            poses_root=poses_root,
                            poses_body=poses_body,
                            poses_right_hand=poses_right_hand,
                            poses_left_hand=poses_left_hand,
                            trans=trans,
                            betas=betas,
                            device=C.device,
                        )

    rhand_thumb_id = smplx_seq.faces[17901][0]
    rhand_fore_id = smplx_seq.faces[17493][0]
    rhand_mid_id = smplx_seq.faces[7051][0]
    rhand_hind_id = smplx_seq.faces[7169][0]
    rhand_small_id = smplx_seq.faces[7285][0]

    lhand_thumb_id = smplx_seq.faces[14168][2]
    lhand_fore_id = smplx_seq.faces[14920][2]
    lhand_mid_id = smplx_seq.faces[15237][2]
    lhand_hind_id = smplx_seq.faces[15724][2]
    lhand_small_id = smplx_seq.faces[14192][2]

    rhand_finger_tips = smplx_seq.vertices[:,[rhand_thumb_id, rhand_fore_id, rhand_mid_id, rhand_hind_id, rhand_small_id],:]
    lhand_finger_tips = smplx_seq.vertices[:,[lhand_thumb_id, lhand_fore_id, lhand_mid_id, lhand_hind_id, lhand_small_id],:]

    outcome = np.concatenate((selected_joints, rhand_finger_tips, lhand_finger_tips), axis=1)[0]

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
        RFL.append(np.linalg.norm(outcome[RHJ[FNSIL[fi]]] - outcome[RHJ[FNEIL[fi]]]))
        RFV.append(outcome[RHJ[FNEIL[fi]]] - outcome[RHJ[FNSIL[fi]]])
        LFL.append(np.linalg.norm(outcome[LHJ[FNSIL[fi]]] - outcome[LHJ[FNEIL[fi]]]))
        LFV.append(outcome[LHJ[FNEIL[fi]]] - outcome[LHJ[FNSIL[fi]]])

    RFRN = [RFV[4*fi] / RFL[4*fi] for fi in range(5)]
    LFRN = [LFV[4*fi] / LFL[4*fi] for fi in range(5)]
    
    for fi in range(5):
        for fni in range(3):
            outcome[RHJ[(fi*4)+2+fni]] = RFL[(fi*4)+1+fni] * RFRN[fi] + outcome[RHJ[(fi*4)+1+fni]]
            outcome[LHJ[(fi*4)+2+fni]] = LFL[(fi*4)+1+fni] * LFRN[fi] + outcome[LHJ[(fi*4)+1+fni]]
    
    return outcome

def extract_finger_tips(label):
    simplx_file = f"E:/HIMO_dataset/data/smplx/{label}.npz"
    simplx_data = np.load(simplx_file, allow_pickle=True)

    nf = simplx_data["body_pose"].shape[0]
    
    poses_root = simplx_data['global_orient']
    poses_body = simplx_data['body_pose'].reshape(nf,-1)
    poses_lhand = simplx_data['lhand_pose'].reshape(nf,-1)
    poses_rhand = simplx_data['rhand_pose'].reshape(nf,-1)
    betas = simplx_data['betas']
    transl = simplx_data['transl']

    poses_root = to_torch(poses_root, dtype=dtype, device=device)
    poses_body = to_torch(poses_body, dtype=dtype, device=device)
    poses_left_hand = to_torch(poses_lhand, dtype=dtype, device=device)
    poses_right_hand = to_torch(poses_rhand, dtype=dtype, device=device)
    betas = to_torch(betas, dtype=dtype, device=device)
    transl = to_torch(transl, dtype=dtype, device=device)

    smplx_layer = SMPLLayer(model_type='smplx',gender='neutral',num_betas=10,device=C.device)

    smplx_seq = SMPLSequence(poses_body=poses_body,
                                smpl_layer=smplx_layer,
                                poses_root=poses_root,
                                betas=betas,
                                trans=transl,
                                poses_left_hand=poses_left_hand,
                                poses_right_hand=poses_right_hand,
                                device=C.device,
                                )

    rhand_thumb_id = smplx_seq.faces[17901][0]
    rhand_fore_id = smplx_seq.faces[17493][0]
    rhand_mid_id = smplx_seq.faces[7051][0]
    rhand_hind_id = smplx_seq.faces[7169][0]
    rhand_small_id = smplx_seq.faces[7285][0]

    lhand_thumb_id = smplx_seq.faces[14168][2]
    lhand_fore_id = smplx_seq.faces[14920][2]
    lhand_mid_id = smplx_seq.faces[15237][2]
    lhand_hind_id = smplx_seq.faces[15724][2]
    lhand_small_id = smplx_seq.faces[14192][2]

    rhand_finger_tips = smplx_seq.vertices[:,[rhand_thumb_id, rhand_fore_id, rhand_mid_id, rhand_hind_id, rhand_small_id],:]
    lhand_finger_tips = smplx_seq.vertices[:,[lhand_thumb_id, lhand_fore_id, lhand_mid_id, lhand_hind_id, lhand_small_id],:]

    return rhand_finger_tips, lhand_finger_tips


C.update_conf({'smplx_models':'E:/HIMO_dataset/body_models'})
dtype = C.f_precision
device = C.device

num_objs = 3

count = 0

label = "S01T001"

subject = label.split('T')[0]

# for file_name in os.listdir(f"../../export/feed_data/new_himo/{num_objs}o/smplx/"):
#     label, _ =os.path.splitext(file_name)
    


    # joints_file = f"joints/{label}.npy"
    # joints_data = np.load(joints_file, allow_pickle=True)

    # joints_lhand = joints_data[:, np.concatenate([range(41,44), range(45,48), range(49,52), range(53,56), range(57,60)]), :]
    # joints_rhand = joints_data[:, np.concatenate([range(17,20), range(21,24), range(25,28), range(29,32), range(33,36)]), :]

simplx_file = f"E:/HIMO_dataset/data/smplx/{label}.npz"
simplx_data = np.load(simplx_file, allow_pickle=True)

nf = simplx_data["body_pose"].shape[0]

betas = simplx_data['betas']
poses_root = simplx_data['global_orient']
poses_body = simplx_data['body_pose'].reshape(nf,-1)
poses_lhand = simplx_data['lhand_pose'].reshape(nf,-1)
poses_rhand = simplx_data['rhand_pose'].reshape(nf,-1)
transl = simplx_data['transl']

poses_root_zero = np.zeros(poses_root.shape)
poses_body_zero = np.zeros(poses_body.shape)
poses_lhand_zero = np.zeros(poses_lhand.shape)
poses_rhand_zero = np.zeros(poses_rhand.shape)
transl_zero = np.zeros(transl.shape)

smplx_layer = SMPLLayer(model_type='smplx',gender='neutral',num_betas=10,device=C.device)

poses_root = to_torch(poses_root_zero, dtype=dtype, device=device)
poses_body = to_torch(poses_body_zero, dtype=dtype, device=device)
poses_left_hand = to_torch(poses_lhand_zero, dtype=dtype, device=device)
poses_right_hand = to_torch(poses_rhand_zero, dtype=dtype, device=device)
betas = to_torch(betas, dtype=dtype, device=device)
transl = to_torch(transl_zero, dtype=dtype, device=device)

verts, joints = smplx_layer(
                poses_root=poses_root,
                poses_body=poses_body,
                poses_left_hand=poses_left_hand,
                poses_right_hand=poses_right_hand,
                betas=betas,
                trans=transl,
            )

extracted_joints = joints[:,np.concatenate([range(0,22),range(25,55)]),:].cpu().numpy()

smplx_seq = SMPLSequence(poses_body=poses_body,
                            smpl_layer=smplx_layer,
                            poses_root=poses_root,
                            betas=betas,
                            trans=transl,
                            poses_left_hand=poses_lhand,
                            poses_right_hand=poses_rhand,
                            device=C.device,
                            )

rhand_thumb_id = smplx_seq.faces[17901][0]
rhand_fore_id = smplx_seq.faces[17493][0]
rhand_mid_id = smplx_seq.faces[7051][0]
rhand_hind_id = smplx_seq.faces[7169][0]
rhand_small_id = smplx_seq.faces[7285][0]

lhand_thumb_id = smplx_seq.faces[14168][2]
lhand_fore_id = smplx_seq.faces[14920][2]
lhand_mid_id = smplx_seq.faces[15237][2]
lhand_hind_id = smplx_seq.faces[15724][2]
lhand_small_id = smplx_seq.faces[14192][2]

rhand_finger_tips = smplx_seq.vertices[0,[rhand_thumb_id, rhand_fore_id, rhand_mid_id, rhand_hind_id, rhand_small_id]]
lhand_finger_tips = smplx_seq.vertices[0,[lhand_thumb_id, lhand_fore_id, lhand_mid_id, lhand_hind_id, lhand_small_id]]

assembled_joints = np.concatenate((extracted_joints[0], rhand_finger_tips, lhand_finger_tips), axis=0)

mesh_to_save = trimesh.Trimesh(vertices=smplx_seq.vertices[0], faces=smplx_seq.faces)
mesh_to_save.export(f"E:/HIMO_dataset/SubjectMesh/{subject}.stl")

joints_to_save = trimesh.points.PointCloud(assembled_joints)
joints_to_save.export(f"E:/HIMO_dataset/SubjectMesh/{subject}_joints.ply")

print(f"Saved mesh of {subject}")

# result = joints[:,np.concatenate([range(0,22),range(25,55)]),:].cpu().numpy()

# has_nan = np.any(np.isnan(result))
# if has_nan:
#     print(label)
#     count = count + 1
# else:
#     output_file_path = f"../../export/feed_data/new_himo/{num_objs}o/computed_joints/{label}.npy" # change this to your path
#     np.save(output_file_path, result)
#     print("save to: ", output_file_path)

# print("Has nan files: ", count)