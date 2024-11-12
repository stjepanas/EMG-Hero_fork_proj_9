import sys
import re
import cv2
from pathlib import Path

import numpy as np

sys.path.append('./')
from emg_hero.defs import MoveConfig

def process_img(img_path):
    img = cv2.imread(img_path.as_posix(), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (x_size, y_size), interpolation = cv2.INTER_AREA)
    new_name = re.sub(r"(\w)([A-Z])", r"\1 \2", img_path.name)
    out_img_file = script_file.parents[1] / 'figures' / 'movements' / new_name
    cv2.imwrite(out_img_file.as_posix(), img)


if __name__ == '__main__':
    move_config = MoveConfig()
    script_file = Path(__file__).resolve()
    image_folder = script_file.parents[2] / 'MiSMT_Env' / 'Images' / 'Fingers'
    images = []

    x_size = 1300
    y_size = 1300
    for move in move_config.movements_wo_rest:
        file_move = move.replace(' ', '').replace('+', '')
        img = cv2.imread((image_folder / (file_move+'.png')).as_posix())
        img = cv2.resize(img, (x_size, y_size), interpolation = cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # make sure background has same color
        brown_lo=np.array([0,0,200])
        brown_hi=np.array([350,10,255])
        mask=cv2.inRange(hsv,brown_lo,brown_hi)
        img[mask>0]=(255,255,255)
        # create alpha
        alpha_channel = np.zeros((img.shape[0], img.shape[1], 1)) + 255
        alpha_channel[mask > 0] = 0

        img = np.concatenate((img, alpha_channel), axis=-1)
        img = cv2.copyMakeBorder(img, 0, 200, 0, 0, cv2.BORDER_CONSTANT, None, (255,255,255))

        out_img_file = script_file.parents[1] / 'figures' / 'movements' / f'{move}.png'
        cv2.imwrite(out_img_file.as_posix(), img)


    # load arm movements
    gross_image_folders = script_file.parents[2] / 'MiSMT_Env' / 'Images' / 'Gross'
    left_folder = gross_image_folders / '1DoF'
    right_folder = gross_image_folders / '1DoFRight'

    for gross_image_folder in [left_folder, right_folder]:
        for img_path in gross_image_folder.glob('*.png'):
            process_img(img_path)

    elbow_flex_file = script_file.parents[1] / 'figures' / 'movements' / 'Flex Elbow.png'
    process_img(elbow_flex_file)
    elbow_extend_file = script_file.parents[1] / 'figures' / 'movements' / 'Extend Elbow.png'
    process_img(elbow_extend_file)
