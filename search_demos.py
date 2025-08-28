import os
import cv2
import numpy as np
# from skimage.metrics import structural_similarity as ssim
import argparse
from openteach.utils.network import ZMQCameraSubscriber
import matplotlib.pyplot as plt

def read_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if success:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def overlay_images(img1, img2, alpha=0.5):
    # Resize to match (if needed)
    # img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    blended = cv2.addWeighted(img1, 1, img2, alpha, 0)
    return blended


def block_miou(imgA, imgB, task=None):
    # hardcode color threshold
    if task == 'stack_blocks_100':
        c1 = [0, 93, 73]
        c2 = [0, 156, 229]
    elif task == 'stack_cups_100':
        c1 = [20, 125, 133]
        c2 = [183, 102, 90]

    imgA = imgA[:, 140:500]
    imgB = imgB[:, 140:500]

    # Convert images to binary masks
    imgA_c1_mask = cv2.inRange(imgA, np.array(c1) - 25, np.array(c1) + 25)
    imgA_c2_mask = cv2.inRange(imgA, np.array(c2) - 25, np.array(c2) + 25)
    imgB_c1_mask = cv2.inRange(imgB, np.array(c1) - 25, np.array(c1) + 25)
    imgB_c2_mask = cv2.inRange(imgB, np.array(c2) - 25, np.array(c2) + 25)
    # plt.imshow(imgA_green_mask, cmap='gray')
    # plt.show(block=True)
    # plt.imshow(imgB_green_mask, cmap='gray')
    # plt.show(block=True)
    # plt.imshow(imgA_cyan_mask, cmap='gray')
    # plt.show(block=True)
    # plt.imshow(imgB_cyan_mask, cmap='gray')
    # plt.show(block=True)
    # breakpoint()

    # Calculate intersection and union
    intersection_c1 = np.logical_and(imgA_c1_mask, imgB_c1_mask)
    intersection_c2 = np.logical_and(imgA_c2_mask, imgB_c2_mask)
    # union_green = np.logical_or(imgA_green_mask, imgB_green_mask)
    # union_cyan = np.logical_or(imgA_cyan_mask, imgB_cyan_mask)
    intersection = np.logical_or(intersection_c1, intersection_c2)
    # union = np.logical_or(union_green, union_cyan)
    # Calculate IoU
    iou = np.sum(intersection)
    return iou, imgB_c1_mask, imgB_c2_mask


def find_most_similar_video(base_dir, reference_image):
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    best_score = -1
    best_video_path = None

    c1_coverage = None
    c2_coverage = None
    overlay = np.zeros_like(reference_image)

    for root, dirs, files in os.walk(base_dir):
        if "cam_2_rgb_video.avi" in files:
            video_path = os.path.join(root, "cam_2_rgb_video.avi")
            first_frame = read_first_frame(video_path)
            if first_frame is not None:
                # breakpoint()
                score, intersection_c1, intersection_c2 = block_miou(reference_image, first_frame, root.split('/')[-2])
                print(f"Compared {video_path} - Similarity Score: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_video_path = video_path
                if c1_coverage is None:
                    c1_coverage = intersection_c1
                    c2_coverage = intersection_c2
                else:
                    c1_coverage = np.logical_or(c1_coverage, intersection_c1)
                    c2_coverage = np.logical_or(c2_coverage, intersection_c2)

                overlay = overlay + first_frame / 100
            else:
                print(f"Failed to read first frame from {video_path}")



    return best_video_path, c1_coverage, c2_coverage, overlay.astype(np.uint8)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the most similar video based on the first frame.")
    parser.add_argument("base_directory", type=str, help="Path to the base directory containing videos.")
    # parser.add_argument("reference_image", type=str, help="Path to the reference image.")
    args = parser.parse_args()

    image_subscriber = ZMQCameraSubscriber(
        host = "143.215.128.151",
        port = "10007",  # 5 - top, 6 - side, 7 - front
        topic_type = 'RGB'
    )

    while True:
        frames = image_subscriber.recv_rgb_image()
        color_frame = frames[0]
        if color_frame is None:
            continue
        break

    base_directory = args.base_directory
    video_path, c1_coverage, c2_coverage, overlay = find_most_similar_video(base_directory, color_frame)
    plt.imshow(c1_coverage, cmap='gray')
    plt.show(block=True)
    plt.imshow(c2_coverage, cmap='gray')
    plt.show(block=True)

    # show the image
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.title(f"{video_path.split('/')[-2]}")
        plt.axis('off')
        plt.show()

