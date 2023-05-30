import sys
from glob import glob
import os
import random
import shutil

if __name__ == "__main__":
    argv = sys.argv[1:]
    dataset_path = argv[0]
    class_paths = glob(dataset_path + "/*/", recursive=False)
    os.mkdir(dataset_path +"/Train/")
    os.mkdir(dataset_path + "/Test/")
    os.mkdir(dataset_path + "/Val/")
    for class_path in class_paths:
        class_name = class_path.split("/")[-2]
        test_path = dataset_path + "/Test/" + class_name + "/"
        train_path = dataset_path + "/Train/" + class_name + "/" 
        val_path = dataset_path + "/Val/" + class_name + "/" 
        os.mkdir(test_path)
        os.mkdir(train_path)
        os.mkdir(val_path)
        all_instances = glob(class_path + "*.png", recursive=False)
        test_val_size = int(0.3 * len(all_instances))
        test_instances = random.sample(all_instances, test_val_size)
        val_instances = random.sample(test_instances, int(0.5*test_val_size))
        test_instances = set(test_instances)
        val_instances = set(val_instances)
        test_instances = test_instances - val_instances
        all_instances = set(all_instances)
        train_instances = (all_instances - test_instances) - val_instances

        for file in test_instances:
            shutil.copy(file, test_path)
        for file in train_instances:
            shutil.copy(file, train_path)
        for file in val_instances:
            shutil.copy(file, val_path)
        
        
