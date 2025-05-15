import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


cv2.setNumThreads(0) 
# 预处理管道
preprocess = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor()
])

def extract_frame(input_video_path: str, target_fold: str, extract_frame_num: int = 10):
    """
    从视频中抽取若干帧并保存为 JPEG。
    """
    # 根据文件名生成 video_id
    ext = os.path.splitext(input_video_path)[1]
    video_id = os.path.basename(input_video_path).replace(ext, '')

    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frame_num = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps * 10))

    for i in range(extract_frame_num):
        frame_idx = int(i * (total_frame_num / extract_frame_num))
        #print(f'[{video_id}] Extract frame {i} ← original frame {frame_idx}, total {total_frame_num}, fps {int(fps)}')
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx - 1, 0))
        success, frame = vidcap.read()
        if not success:
            print(f'[{video_id}] Warning: failed to read frame {frame_idx}')
            continue

        # BGR → RGB → PIL → Tensor
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        image_tensor = preprocess(pil_im)

        # 保存路径
        frame_dir = os.path.join(target_fold, f'frame_{i}')
        os.makedirs(frame_dir, exist_ok=True)
        out_path = os.path.join(frame_dir, f'{video_id}.jpg')
        save_image(image_tensor, out_path)

def _worker(args):
    """Pool 的包装函数，捕获异常并打印日志"""
    input_path, target_fold = args
    try:
        extract_frame(input_path, target_fold)
    except Exception as e:
        print(f'[ERROR] {input_path}: {e}')

if __name__ == "__main__":
    parser = ArgumentParser(description="并行抽取视频帧")
    parser.add_argument(
        "-input_file_list", type=str,
        default='/data/wanglinge/project/cav-mae/src/data/info/k700_train.csv',
        help="包含视频路径的 CSV 文件（单列，无表头或已跳过表头）"
    )
    parser.add_argument(
        "-target_fold", type=str,
        default='/data/wanglinge/project/cav-mae/src/data/k700/train',
        help="保存帧的目标文件夹"
    )
    parser.add_argument(
        "-num_workers", type=int, default=64,
        help="并行进程数，默认为 CPU 核心数"
    )
    args = parser.parse_args()

    input_filelist = np.loadtxt(args.input_file_list, dtype=str, delimiter=',')
    input_filelist = input_filelist
    tasks = [(path, args.target_fold) for path in input_filelist]
    n_workers = args.num_workers or cpu_count()
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            result = future.result()
            results.append(result)


            
    # print(f'[INFO] Total {len(tasks)} videos | Using {n_workers} workers')

    # # with Pool(processes=n_workers) as pool:
    # #     pool.map(_worker, tasks)

    # with Pool(processes=n_workers) as pool:
    #     results = []
    #     for _ in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), desc="Processing videos"):
    #         results.append(_)
    
    # print(f'[INFO] Done. {len([r for r in results if r is not None])} videos processed successfully.')