import json
import os

info_dir = "/data/wanglinge/project/cav-mae/src/data/info/k700/k700_val_valid.json"
output_dir = "/data/wanglinge/project/cav-mae/src/data/info/k700/k700_val_valid_1.json"
output_dir_2 = "/data/wanglinge/project/cav-mae/src/data/info/k700/k700_val_valid_2.json"

def split_dataset(info_dir, output_dir, output_dir_2):
    with open(info_dir, 'r') as f:
        raw_data = json.load(f)

    data = raw_data['data']
    # Split the dataset into two halves
    half_size = len(data) // 5
    data_1 = data[:half_size]
    data_2 = data[half_size:]

    # Create new dictionaries for each half
    new_data_1 = {
        'data': data_1,
    }
    new_data_2 = {
        'data': data_2,
    }
    # Write the new datasets to output files
    with open(output_dir, 'w') as f:
        json.dump(new_data_1, f, indent=4)  
    with open(output_dir_2, 'w') as f:
        json.dump(new_data_2, f, indent=4)

if __name__ == "__main__":
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))
    if not os.path.exists(os.path.dirname(output_dir_2)):
        os.makedirs(os.path.dirname(output_dir_2))
    
    split_dataset(info_dir, output_dir, output_dir_2)
    print(f"Dataset split completed. Outputs saved to {output_dir} and {output_dir_2}.")