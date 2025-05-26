import json


input_path = '/data/wanglinge/project/cav-mae/src/data/info/as/data/eval_segments_valid.json'
output_path1 = '/data/wanglinge/project/cav-mae/src/data/info/as/data/eval_segments_valid_1.json'
output_path2 = '/data/wanglinge/project/cav-mae/src/data/info/as/data/eval_segments_valid_2.json'

with open(input_path, 'r') as f:
    data_json = json.load(f)

nums = len(data_json['data'])
print(f'原始数据集长度: {nums}')
data_1 = data_json['data'][:nums//5]
data_2 = data_json['data'][nums//5:]

with open(output_path1, 'w') as f:
    json.dump({'data': data_1}, f, indent=2)
with open(output_path2, 'w') as f:
    json.dump({'data': data_2}, f, indent=2)

