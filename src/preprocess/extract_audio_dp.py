import os
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def extract_audio(task):
    input_f, target_fold = task
    input_f = input_f[0]
    ext = input_f.split('.')[-1]
    video_id = os.path.basename(input_f)[:-len(ext) - 1]

    output_f_1 = os.path.join(target_fold, f'{video_id}_intermediate.wav')
    output_f_2 = os.path.join(target_fold, f'{video_id}.wav')

    # Step 1: extract and resample audio with ffmpeg
    cmd1 = f'ffmpeg -i "{input_f}" -vn -ar 16000 "{output_f_1}" -y -loglevel error'
    ret1 = os.system(cmd1)

    if ret1 != 0 or not os.path.exists(output_f_1):
        return False  # Failed

    # Step 2: extract 1st channel with sox
    cmd2 = f'sox "{output_f_1}" "{output_f_2}" remix 1'
    ret2 = os.system(cmd2)

    # Clean up intermediate file
    if os.path.exists(output_f_1):
        os.remove(output_f_1)

    return ret2 == 0 and os.path.exists(output_f_2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-threaded audio extractor')
    parser.add_argument("-input_file_list", type=str, default='/data/wanglinge/project/cav-mae/src/data/info/k700/k700_val.csv', help="CSV file: one video path per line")
    parser.add_argument("-target_fold", type=str, default='/data/wanglinge/dataset/k700/audio', help="Folder to store output audio")
    parser.add_argument("-num_workers", type=int, default=16, help="Number of threads to use")
    args = parser.parse_args()

    input_filelist = np.loadtxt(args.input_file_list, delimiter=',', dtype=str)
    input_filelist = input_filelist
    os.makedirs(args.target_fold, exist_ok=True)

    # Build tasks list
    tasks = [(path, args.target_fold) for path in input_filelist]

    # Run with ThreadPoolExecutor
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(extract_audio, task) for task in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Extracting audio"):
            success_count += int(f.result() is True)

    print(f"[INFO] Done. Successfully processed {success_count}/{len(tasks)} videos.")
