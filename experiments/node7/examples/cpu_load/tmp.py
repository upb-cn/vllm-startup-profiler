import os
import shutil

workload = "90"

for i in range(1, 6):
    src_dir_path = f"./iterations{workload}/{i}"
    dst_dir_path = f"./iterations/{i}"
    for file in os.listdir(src_dir_path):
        
        if not file.startswith("output_"):
            continue
        
        old_filepath = os.path.join(src_dir_path, file)
        
        filename = file.split('.txt')[0]
        new_filename = filename + f"_workload_{workload}.txt"
        new_filepath = os.path.join(src_dir_path, new_filename)

        os.rename(old_filepath, new_filepath)
        
        
        
        dst_filepath = os.path.join(dst_dir_path, new_filename)
        shutil.move(new_filepath, dst_filepath)