import json
import os

# 输入输出 JSON 文件路径
input_json = "/data/wanglinge/project/cav-mae/src/data/info/as/data/eval_segments.json"
output_json = "/data/wanglinge/project/cav-mae/src/data/info/as/data/eval_segments_valid.json"

# 加载原始 JSON
with open(input_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 过滤数据：只保留文件存在的条目
filtered_data = []
for entry in data["data"]:
    wav_path = entry["wav"]
    if os.path.exists(wav_path):
        filtered_data.append(entry)
    else:
        print(f"[Warning] File not found: {wav_path}")

# 保存过滤后的 JSON
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump({"data": filtered_data}, f, indent=2)

print(f"\nDone. Valid entries: {len(filtered_data)}. Saved to: {output_json}")
