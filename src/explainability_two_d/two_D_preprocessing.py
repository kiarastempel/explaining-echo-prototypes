from pathlib import Path
from PIL import Image
import cv2
import pandas as pd
import tensorflow as tf
import random
import os
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data',
                        help="Directory with the echocardiograms.")
    parser.add_argument('-o', '--output_directory',
                        help="Directory to save still images in.")
    parser.add_argument('-m', '--metadata_filename', default='FileList.csv',
                        help="Name of the metadata file.")
    parser.add_argument('-v', '--volumes_filename', default='VolumeTracings.csv',
                        help="Name of the volume tracings file.")
    args = parser.parse_args()

    avi_directory = Path(args.input_directory, 'Videos')
    if args.output_directory is None:
        output_directory = Path(args.input_directory, 'still_images')
    else:
        output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.input_directory, args.metadata_filename)
    volumes_path = Path(args.input_directory, args.volumes_filename)

    tf.random.set_seed(5)
    random.seed(5)

    volume_tracings_data_frame = get_volume_tracings(metadata_path, volumes_path, output_directory)
    create_still_images(volume_tracings_data_frame, avi_directory, output_directory)


def create_still_images(volume_tracings_data_frame, avi_directory, output_directory):
    for _, row in volume_tracings_data_frame.T.iteritems():
        file_path = os.path.join(avi_directory, row['FileName'])
        frame_position = row['Frame']
        # print('File to convert:', file_path)
        # print('Frame position', frame_position)
        video = cv2.VideoCapture(str(file_path))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ret, frame = video.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video.release()
        # print('Frame', frame)
        # print('Shape', frame.shape)

        # save frame as png file
        image = Image.fromarray(frame)
        output_path = Path(output_directory, row['Image_FileName'])
        image.save(output_path)
        # plt.figure(figsize=(5, 5))
        # plt.axis('off')
        # plt.imshow(frame)
        # plt.savefig(output_path)
        # plt.close()


def get_volume_tracings(metadata_path, volumes_path, output_directory):
    file_list_data_frame = pd.read_csv(metadata_path, sep=',', decimal='.')[['FileName', 'ESV', 'EDV']]
    file_list_data_frame['FileName'] = file_list_data_frame['FileName'].astype(str) + '.avi'
    volume_tracings_data_frame = pd.read_csv(volumes_path, sep=',', decimal='.')[['FileName', 'Frame']]
    volume_tracings_data_frame = volume_tracings_data_frame.groupby(
        ['FileName', 'Frame']).head(1).reset_index(drop=True)
    esv_tracings_data_frame = volume_tracings_data_frame.loc[
        volume_tracings_data_frame.groupby('FileName')['Frame'].idxmin()] \
        .merge(file_list_data_frame[['FileName', 'ESV']], on='FileName',
               how='inner') \
        .rename(columns={'ESV': 'Volume'})
    edv_tracings_data_frame = volume_tracings_data_frame.loc[
        volume_tracings_data_frame.groupby('FileName')['Frame'].idxmax()] \
        .merge(file_list_data_frame[['FileName', 'EDV']], on='FileName',
               how='inner') \
        .rename(columns={'EDV': 'Volume'})
    volume_tracings_data_frame = pd.concat(
        [esv_tracings_data_frame, edv_tracings_data_frame]).reset_index(drop=True)
    volume_tracings_data_frame['Image_FileName'] = \
        volume_tracings_data_frame['FileName'].str.replace(r'.avi$', '') + '_' \
        + volume_tracings_data_frame['Frame'].astype(str) + '.png'

    # split into train, validation and test data
    volume_tracings_data_frame['Split'] = ''
    for i, row in volume_tracings_data_frame.T.iteritems():
        r = random.uniform(0, 1)
        if r < 0.75:  # 75% train
            volume_tracings_data_frame.at[i, 'Split'] = 'TRAIN'
        elif r < 0.875:  # 12.5% test
            volume_tracings_data_frame.at[i, 'Split'] = 'TEST'
        else:  # 12.5% validation
            volume_tracings_data_frame.at[i, 'Split'] = 'VAL'
    volume_tracings_data_frame.to_csv(Path(output_directory, 'FrameVolumes.csv'), index=False)
    return volume_tracings_data_frame


if __name__ == '__main__':
    main()
