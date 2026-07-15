
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# def read_imgs(img_list):
#     frames = []
#     logger.info('reading images...')
#     for img_path in tqdm(img_list):
#         frame = cv2.imread(img_path)
#         frames.append(frame)
#     return frames

def read_imgs(img_list):
    def load_image(index, img_path):
        return index, cv2.imread(img_path)

    frames = [None] * len(img_list)  # Initialize a list with the same length as img_list
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_image, idx, img_path): idx for idx, img_path in enumerate(img_list)}
        for future in tqdm(as_completed(futures), total=len(img_list)):
            idx, img = future.result()
            frames[idx] = img
    return frames


def remove_legacy_livetalking_watermark(frames):
    """Remove the legacy top-left watermark from previously generated avatars."""
    for index, frame in enumerate(frames):
        if frame is None:
            continue

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.putText(
            mask,
            "LiveTalking",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            255,
            1,
        )

        # Old generators wrote solid RGB(128, 128, 128) pixels into PNG frames.
        legacy_pixels = cv2.inRange(frame, (126, 126, 126), (130, 130, 130))
        mask = cv2.bitwise_and(mask, legacy_pixels)
        if cv2.countNonZero(mask) >= 8:
            frames[index] = cv2.inpaint(frame, mask, 2, cv2.INPAINT_TELEA)

    return frames

def mirror_index(size, index):
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1
