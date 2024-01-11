import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import math
import argparse

class VideoFrame():
    def __init__(self, video_name, csv_name):
        self.cap = cv2.VideoCapture(video_name)
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_df =  pd.read_csv(csv_name, header=None)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        channels = 3
        self.fake_frame = 255*np.ones((height,width,channels), dtype=np.uint8)

        self.surplus_fps = 0.0
        self.num_copies = 0
        self.orig_frame_no = 0

        return

    def __del__(self):
        cv2.destroyAllWindows()

    def get_fake_frame(self):
        return self.fake_frame

    def read_next(self):
        ret = True
        if self.num_copies == 0:
            ret, self.current_frame = self.cap.read()
            self.orig_frame_no +=1
            if ret:
                self.num_copies = 1 + self.calculate_duplicate_frames()

        self.num_copies -=1
        return ret, self.current_frame

    def calculate_duplicate_frames(self) -> int:
        c_frame = self.orig_frame_no
        if c_frame <= 1:
            return 0

        sec_per_frame = self.frame_df[self.frame_df[0] == c_frame].reset_index(drop=True)[1][0]
        val = round(self.FPS * sec_per_frame, 4)
        int_duplicate_frames= math.floor(val)
        self.surplus_fps += val - int_duplicate_frames

        if self.surplus_fps >=1:
            int_x = math.floor(self.surplus_fps)
            self.surplus_fps = self.surplus_fps - int_x
            int_duplicate_frames += int_x

        return int_duplicate_frames

class PostProcessVideo():
    def __init__(self, video_names, csv_names, output_name):
        self.video_one = VideoFrame(video_names[0], csv_names[0])
        self.video_two = VideoFrame(video_names[1], csv_names[1])
        self.FPS = self.video_one.FPS
        self.min_total_frames = min(self.video_one.total_frames, self.video_two.total_frames)
        self.LEN_VIDEOS = len(video_names)

        self.out = cv2.VideoWriter()
        self.output_name = output_name
        is_first = True
        skip_frame1 = False

        while True:
            ret1 = False
            if not skip_frame1:
                ret1, frame1 = self.video_one.read_next()

            skip_frame1 = not ret1

            if skip_frame1:
                frame1 = self.video_one.get_fake_frame()

            ret2, frame2 = self.video_two.read_next()


            if not ret2:
                break

            self.store_frame(frame1, frame2, is_first)
            is_first = False

    def store_frame(self,frame1, frame2, is_first=False):
        if is_first:
            both_frames = cv2.hconcat([frame1, frame2])
            (height, width, _) = both_frames.shape
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            FILE_OUTPUT = self.output_name
            self.out = cv2.VideoWriter(FILE_OUTPUT, fourcc, self.FPS, (width, height))
        else:
            both_frames = cv2.hconcat([frame1, frame2])
            self.out.write(both_frames)
        return

if __name__ == '__main__':
    # video_file_name='face8.mp4'
    # result_dir='results'

    parser = argparse.ArgumentParser()
    parser.add_argument('--video-name', default=None, type=lambda s: s.lower(),
                        help='video name')

    parser.add_argument('--result-dir', default='results', type=lambda s: s.lower(),
                        help='directory containing test images')
    args = parser.parse_args()

    video_file_name = args.video_name
    result_dir = args.result_dir

    mp4_file = video_file_name
    video_name,video_ext = os.path.splitext(video_file_name)
    output_name=os.path.join(result_dir,mp4_file)
    auto_split_csv = 'auto_split_timestamp_{}.csv'.format(video_name)
    cloud_only_csv = 'cloud_only_timestamp_{}.csv'.format(video_name)
    csv_name_list = [os.path.join(result_dir, auto_split_csv), os.path.join(result_dir, cloud_only_csv)]

    auto_split_mp4 = 'auto_split_{}.mp4'.format(video_name)
    cloud_only_mp4 = 'cloud_only_{}.mp4'.format(video_name)
    video_name_list = [os.path.join(result_dir, auto_split_mp4), os.path.join(result_dir, cloud_only_mp4)]

    PostProcessVideo(video_name_list, csv_name_list,output_name)
