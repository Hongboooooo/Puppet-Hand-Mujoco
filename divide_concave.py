import coacd
import trimesh
from pathlib import Path
import os


# object_name = "Mug"

for obj_name in os.listdir("E:/HIMO_DATASET/data/object_mesh/"):
    object_name, _ = os.path.splitext(obj_name)
    print(f"Processing {object_name}")

    input_file = f"E:/HIMO_DATASET/data/object_mesh/{object_name}.obj"

    output_file = Path(f"E:/HIMO_DATASET/data/object_mesh_divided/{object_name}")

    output_file.mkdir(parents=True, exist_ok=True)

    import_mesh = trimesh.load(input_file, force="mesh")

    mesh_to_divide = coacd.Mesh(import_mesh.vertices, import_mesh.faces)

    parts = coacd.run_coacd(mesh_to_divide)

    for pi in range(len(parts)):

        mesh_to_save = trimesh.Trimesh(vertices=parts[pi][0], faces=parts[pi][1])
        mesh_to_save.export(f"{output_file}/part{pi}.stl")
        print(f"Saved part{pi}")
