
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class testLoader(BaseLoader):
    """The data loader for the test dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes an test dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- subject1/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |   |-- subject2/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |...
                     |   |-- subjectn/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For test dataset)."""
        print('getting raw data')
        data_dirs = glob.glob(data_path + os.sep + "subject*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search(
            'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        try:
            filename = os.path.split(data_dirs[i]['path'])[-1]
            saved_filename = data_dirs[i]['index']
            print(f'Test loader processing: {saved_filename}')

            # Read Frames
            if 'None' in config_preprocess.DATA_AUG:
                # Utilize dataset-specific function to read video
                frames = self.read_video(
                    os.path.join(data_dirs[i]['path'],"vid.avi"),
                    width=520,
                    height=520)
            elif 'Motion' in config_preprocess.DATA_AUG:
                # Utilize general function to read video in .npy format
                frames = self.read_npy_video(
                    glob.glob(os.path.join(data_dirs[i]['path'],'*.npy')))
            else:
                raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

            # Read Labels
            if config_preprocess.USE_PSUEDO_PPG_LABEL:
                bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
            else:
                bvps = self.read_wave(
                    os.path.join(data_dirs[i]['path'],"ground_truth.txt"))
            
            # Common fix: ensure label length matches frame count before chunking
            if len(bvps) != frames.shape[0]:
                bvps = BaseLoader.resample_ppg(bvps, frames.shape[0])
                
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
            file_list_dict[i] = input_name_list
            print(f"Successfully finished processing {saved_filename}")

        except Exception as e:
            print(f"\n[ERROR] Subprocess {i} failed for {data_dirs[i]['path']}: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def read_video(video_file, width=128, height=128):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        total_frames = int(VidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(VidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(VidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Reading {video_file}: {total_frames} frames. Resizing from {orig_width}x{orig_height} to {width}x{height}")
        
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)

        # Optimization: Pre-allocate array to prevent OOM crash during conversion
        if total_frames > 0:
            frames = np.zeros((total_frames, height, width, 3), dtype=np.uint8)
            for i in range(total_frames):
                success, frame = VidObj.read()
                if not success:
                    frames = frames[:i] # Truncate if codec lied about frame count
                    break
                resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                frames[i] = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        else:
            # Fallback if frame count is unknown
            frames_list = []
            while True:
                success, frame = VidObj.read()
                if not success: break
                resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                frames_list.append(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            frames = np.asarray(frames_list)

        VidObj.release()
        print(f'Finished reading video. Array shape: {frames.shape}')
        return frames

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)
