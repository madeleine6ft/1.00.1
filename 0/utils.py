import os

def get_target_name(path_data):
    target_name_list = []
    for a, b, c in os.walk(path_data):
        for file_name in c:
            if file_name.find("csv") != -1:
                target_name_list.append(file_name)
    return target_name_list