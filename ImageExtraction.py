import os
import shutil
import string

def files_in_dir(dir):
  files = []
  for file in os.listdir(dir):  # os.listdir creates a list of all files and directories in a directory
    files.append(file)
  return files


base_image_path = "C:\\Users\\zglas\\Pictures\\BirdProject"

'''# Only getting MN Birds
bird_list_file = "C:\\Users\\zglas\\Documents\\Minnesota_CT_Birds.txt"
birds_to_keep = []
with open(bird_list_file) as f:
    for line in f:
        birds_to_keep.append(line.split()[0])

print(birds_to_keep)
all_birds = files_in_dir(base_image_path)
print(all_birds)
for bird in all_birds:
    keep = False
    for entry in birds_to_keep:
        if entry in bird:
            keep = True
            break
    if not keep:
        fl = 0
        while bird[fl] not in string.ascii_letters + string.digits:
            fl += 1
        shutil.rmtree(base_image_path + "\\" + bird)

'''
train_path = base_image_path + "\\train"
validation_path = base_image_path + "\\validation"

for folder in files_in_dir(train_path):
    validation_dest_path = validation_path + "\\" + folder
    os.mkdir(validation_dest_path)

    src_path = train_path + "\\" + folder
    pictures = files_in_dir(src_path)
    for pic in pictures[:len(pictures)//5]:
        shutil.move(src_path + "\\" + pic, validation_dest_path)

for folder in files_in_dir(train_path):
    fl = 0
    while folder[fl] not in string.ascii_letters:
        fl += 1
    os.rename(train_path + "\\" + folder, train_path + "\\" + folder[fl:])

for folder in files_in_dir(validation_path):
    fl = 0
    while folder[fl] not in string.ascii_letters:
        fl += 1
    os.rename(validation_path + "\\" + folder, validation_path + "\\" + folder[fl:])


