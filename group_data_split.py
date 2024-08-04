import face_recognition
import os
from pathlib import Path
from shutil import copyfile
import random
import math

# Define directories and paths
cwd = os.getcwd()
p = os.path.join(cwd, 'collected_data_double')
train_results_path = os.path.join(cwd, 'results_train_double')
test_results_path = os.path.join(cwd, 'results_test_double')
validate_results_path = os.path.join(cwd, 'results_validate_double')

train_split = 0.8
test_split = 0.1
validate_split = 0.1

encodings = {}

# Function to process each file
def process_file(filepath, filename):
    img = face_recognition.load_image_file(filepath)
    fe = face_recognition.face_encodings(img)
    
    # If no face encodings found, return
    if not fe:
        return
    
    fe = fe[0]  # Take the first face encoding
    
    action_taken = False
    curr_image_cluster_id = None
    
    # Iterate through existing clusters
    for cluster_id, cluster_encodings in encodings.items():
        results = face_recognition.compare_faces(cluster_encodings, fe)
        
        # If the current image matches all the encodings of the encodings we have for that cluster,
        # we add the current image to that cluster
        if all(results):
            curr_image_cluster_id = cluster_id
            encodings[cluster_id].append(fe)
            action_taken = True
            break
    
    # If the current image was not added to any cluster we have, create a new cluster
    if not action_taken:
        curr_image_cluster_id = f"cluster_{len(encodings) + 1}"
        encodings[curr_image_cluster_id] = [fe]
    
    # Determine which results directory to use based on random split
    rand_split = random.random()
    if rand_split < train_split:
        results_path = train_results_path
    elif rand_split < train_split + test_split:
        results_path = test_results_path
    else:
        results_path = validate_results_path
    
    # Create a directory for the current cluster within the appropriate results directory
    curr_cluster = os.path.join(results_path, curr_image_cluster_id)
    curr_cluster_dir = Path(curr_cluster)
    curr_cluster_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the file to the corresponding cluster's folder
    copyfile(filepath, os.path.join(curr_cluster_dir, filename))

# Iterate through all files in the 'collected_data' directory
for subdir, dirs, files in os.walk(p):
    for file in files:
        filepath = os.path.join(subdir, file)
        print("File: %s" % filepath)
        process_file(filepath, file)

print("Processing complete.")