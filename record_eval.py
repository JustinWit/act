from openteach.utils.network import ZMQCameraSubscriber
import cv2
import threading
from time import sleep
import datetime
import os

class RecordEval():
    def __init__(self, ckpt_name):
        self.image_list = []
        self.record = False
        self.ckpt_name = ckpt_name.split('.')[0]
        self.image_subscriber = ZMQCameraSubscriber(
                                    host = "143.215.128.151",
                                    port = "10005",  # 5 - top, 6 - side, 7 - front
                                    topic_type = 'RGB'
                                )
        self.record_thread = threading.Thread(target=self.save_frames)

    def save_frames(self):
        print("Recording")
        while self.record:
            frames = self.image_subscriber.recv_rgb_image()
            color_frame = frames[0]
            if color_frame is None:
                continue

            # process and store image
            self.image_list.append(color_frame)

    def start(self):
        self.record = True
        self.record_thread.start()

    def stop(self, root_path):
        self.record = False
        self.record_thread.join()
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        filename = f'{datetime.datetime.now().strftime(f"{self.ckpt_name}_%Y-%m-%d_%H-%M-%S")}.mp4'
        save_path = os.path.join('.', root_path, filename)
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, self.image_list[0].shape[:2][::-1])
        print(len(self.image_list))
        for image in self.image_list:
            out.write(image)
        print(f'Saved video to: {save_path}')


if __name__ == '__main__':
    re = RecordEval()
    re.start()
    sleep(5)
    re.stop('.')