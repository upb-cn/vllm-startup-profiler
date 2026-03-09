import os
import shutil


# for i in range (1, 6):
#     dir_path = os.path.join("iterations_new_default", str(i))
    
#     for file in os.listdir(dir_path):
#         if not file.endswith(".txt"):
#             continue
        
#         old_file_path = os.path.join(dir_path, file)
#         old_file_name = file.split(".txt")[0]
        
#         new_file_name = old_file_name + "_default.txt"
#         new_file_path = os.path.join(dir_path, new_file_name)
        
#         os.rename(old_file_path, new_file_path)


for i in range (1, 6):
    old_dir_path = os.path.join("iterations_old", str(i))
    new_dir_path = os.path.join("iterations", str(i))
    
    for file in os.listdir(old_dir_path):
        if not file.endswith(".txt"):
            continue
        
        if "_container" in file:
            
            old_file_path = os.path.join(old_dir_path, file)        
            new_file_path = os.path.join(new_dir_path, file)
            
            shutil.copy(old_file_path, new_file_path)