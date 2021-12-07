from pathlib import Path
from PIL import Image
from shapely.geometry import Polygon
import cv2
import pandas as pd
import tensorflow as tf
import random
import os
import argparse


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

    delete_incorrect_data(metadata_path, volumes_path)
    volume_tracings_data_frame = get_volume_tracings(metadata_path, volumes_path, output_directory)
    create_still_images(volume_tracings_data_frame, avi_directory, output_directory)


def create_still_images(volume_tracings_data_frame, avi_directory, output_directory):
    for _, row in volume_tracings_data_frame.T.iteritems():

        file_path = os.path.join(avi_directory, row['FileName'])
        frame_position = row['Frame']
        video = cv2.VideoCapture(str(file_path))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ret, frame = video.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video.release()

        # save frame as png file
        image = Image.fromarray(frame)
        output_path = Path(output_directory, row['ImageFileName'])
        image.save(output_path)


def get_volume_tracings(metadata_path, volumes_path, output_directory):
    file_list_data_frame = pd.read_csv(metadata_path, sep=',', decimal='.')[['FileName', 'ESV', 'EDV']]
    file_list_data_frame['FileName'] = file_list_data_frame['FileName'].astype(str) + '.avi'
    volume_tracings_data_frame = pd.read_csv(volumes_path, sep=',', decimal='.')

    # save all 21 coordinate pairs of left ventricle as two lists: X, Y
    coordinates = volume_tracings_data_frame.groupby(['FileName', 'Frame'])\
        .agg({'X1': lambda x: x.tolist()[1:],
              'Y1': lambda x: x.tolist()[1:],
              'X2': lambda x: x.tolist()[1:][::-1],
              'Y2': lambda x: x.tolist()[1:][::-1]})
    # concatenate X1 + reversed X2, Y1 + reversed Y2 to get a list of all pairs
    coordinates['X'] = coordinates['X1'] + coordinates['X2']
    coordinates['Y'] = coordinates['Y1'] + coordinates['Y2']

    volume_tracings_data_frame = volume_tracings_data_frame.groupby(
        ['FileName', 'Frame']).head(1).reset_index(drop=True)

    # merge df containing X- and Y-lists and calculate axis length and area
    volume_tracings_data_frame = volume_tracings_data_frame \
        .merge(coordinates[['X', 'Y']], on=['FileName', 'Frame'], how='inner')
    volume_tracings_data_frame = calc_axis_length(volume_tracings_data_frame)
    volume_tracings_data_frame = calc_poly_area(volume_tracings_data_frame)

    esv_tracings_data_frame = volume_tracings_data_frame.loc[
        volume_tracings_data_frame.groupby('FileName')['AxisLength'].idxmin()] \
        .merge(file_list_data_frame[['FileName', 'ESV']], on='FileName',
               how='inner') \
        .rename(columns={'ESV': 'Volume'})
    esv_tracings_data_frame['ESV/EDV'] = 'ESV'
    edv_tracings_data_frame = volume_tracings_data_frame.loc[
        volume_tracings_data_frame.groupby('FileName')['AxisLength'].idxmax()] \
        .merge(file_list_data_frame[['FileName', 'EDV']], on='FileName',
               how='inner') \
        .rename(columns={'EDV': 'Volume'})
    edv_tracings_data_frame['ESV/EDV'] = 'EDV'
    volume_tracings_data_frame = pd.concat(
        [esv_tracings_data_frame, edv_tracings_data_frame]).reset_index(drop=True)
    volume_tracings_data_frame = volume_tracings_data_frame.sort_values(
        by=['FileName', 'Frame']).reset_index(drop=True)
    volume_tracings_data_frame['ImageFileName'] = \
        volume_tracings_data_frame['FileName'].str.replace(r'.avi$', '') + '_' \
        + volume_tracings_data_frame['Frame'].astype(str) + '.png'
    print(volume_tracings_data_frame[['FileName', 'Frame', 'AxisLength', 'ESV/EDV', 'PolyArea']])

    # split into train, validation and test data
    volume_tracings_data_frame['Split'] = ''
    volume_tracings_data_frame = volume_tracings_data_frame.sort_values(by=['FileName', 'Frame']).reset_index(drop=True)
    counter_smaller = 0
    counter_bigger = 0
    for i, row in volume_tracings_data_frame.T.iteritems():
        # assign both frames of same image to same dataset
        if i % 2 == 0:
            if volume_tracings_data_frame.at[i, 'AxisLength'] < volume_tracings_data_frame.at[i + 1, 'AxisLength']:
                counter_smaller += 1  # first ESV, then EDV
            else:
                counter_bigger += 1  # first EDV, then ESV
            if (volume_tracings_data_frame.at[i, 'AxisLength'] < volume_tracings_data_frame.at[i + 1, 'AxisLength']
                    and volume_tracings_data_frame.at[i, 'Volume'] > volume_tracings_data_frame.at[i + 1, 'Volume']) \
                    or (volume_tracings_data_frame.at[i, 'AxisLength'] > volume_tracings_data_frame.at[i + 1, 'AxisLength']
                        and volume_tracings_data_frame.at[i, 'Volume'] < volume_tracings_data_frame.at[i + 1, 'Volume']):
                print('Wrong assigment at index', i)
            r = random.uniform(0, 1)
            if r < 0.75:  # 75% train
                volume_tracings_data_frame.at[i, 'Split'] = 'TRAIN'
                volume_tracings_data_frame.at[i + 1, 'Split'] = 'TRAIN'
            elif r < 0.875:  # 12.5% test
                volume_tracings_data_frame.at[i, 'Split'] = 'TEST'
                volume_tracings_data_frame.at[i + 1, 'Split'] = 'TEST'
            else:  # 12.5% validation
                volume_tracings_data_frame.at[i, 'Split'] = 'VAL'
                volume_tracings_data_frame.at[i + 1, 'Split'] = 'VAL'
    print(volume_tracings_data_frame.groupby('Split').count())
    print("bigger:", counter_bigger)
    print("smaller:", counter_smaller)
    volume_tracings_data_frame.to_csv(Path(output_directory, 'FrameVolumes.csv'), index=False)
    return volume_tracings_data_frame


def delete_incorrect_data(metadata_path, volumes_path):
    file_list_data_frame = pd.read_csv(metadata_path, sep=',', decimal='.')
    volume_tracings_data_frame = pd.read_csv(volumes_path, sep=',', decimal='.')
    count = volume_tracings_data_frame.groupby(['FileName', 'Frame']).count()
    incorrect_files_frame = count[(count < 21).any(1)]
    # drop file ending
    incorrect_files = list(set([x[0][:-4] for x in incorrect_files_frame['X1'].to_dict().keys()]))
    # keep file ending (avi)
    incorrect_files_avi = list(set([x[0] for x in incorrect_files_frame['X1'].to_dict().keys()]))
    print('Number of files having too many coordinates:', len(incorrect_files))

    file_list_data_frame = file_list_data_frame.set_index('FileName').drop(incorrect_files)
    file_list_data_frame.to_csv(metadata_path)
    volume_tracings_data_frame = volume_tracings_data_frame.set_index('FileName').drop(incorrect_files_avi)
    volume_tracings_data_frame.to_csv(volumes_path)
    incorrect_files_frame.to_csv('incorrect.csv')
    print('saved')


def calc_axis_length(df):
    df['AxisLength'] = ((df.X1.sub(df.X2) ** 2).add(
        df.Y1.sub(df.Y2) ** 2)) ** 0.5
    return df


def calc_poly_area(df):
    df['PolyArea'] = 0.0
    for i, row in df.T.iteritems():
        polygon = Polygon(zip(row['X'], row['Y']))
        df.at[i, 'PolyArea'] = Polygon(zip(row['X'], row['Y'])).area
    return df


if __name__ == '__main__':
    main()
