import argparse
import pandas
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data',
                        help='Directory with the echocardiograms.')
    parser.add_argument('-s', '--still_images',
                        default='../../data/still_images',
                        help='Directory containing the still images.')
    parser.add_argument('-m', '--metadata_filename', default='FileList.csv',
                        help='Name of the metadata file.')
    parser.add_argument('-fv', '--frame_volumes_filename',
                        default='FrameVolumes.csv',
                        help='Name of the file containing frame volumes.')
    args = parser.parse_args()

    # distribution of data
    frame_volumes_path = Path(args.still_images, args.frame_volumes_filename)
    volume_tracings_data_frame = pd.read_csv(frame_volumes_path)
    # get test data
    volume_tracings_data_frame_test = volume_tracings_data_frame[
        volume_tracings_data_frame.Split == 'TEST']
    print('Mean volume of test data',
          volume_tracings_data_frame_test[['Volume']].mean(axis=0))
    volume_tracings_data_frame_test = volume_tracings_data_frame_test.rename(
        columns={'Volume': 'Test'})
    # get validation data
    volume_tracings_data_frame_val = volume_tracings_data_frame[
        volume_tracings_data_frame.Split == 'VAL']
    print('Mean volume of validation data',
          volume_tracings_data_frame_val[['Volume']].mean(axis=0))
    volume_tracings_data_frame_val = volume_tracings_data_frame_val.rename(
        columns={'Volume': 'Validation'})
    # get training data
    volume_tracings_data_frame_train = volume_tracings_data_frame[
        volume_tracings_data_frame.Split == 'TRAIN']
    print('Mean volume of train data',
          volume_tracings_data_frame_train[['Volume']].mean(axis=0), '\n')
    volume_tracings_data_frame_train = volume_tracings_data_frame_train.rename(
        columns={'Volume': 'Train'})
    # plot volume distributions
    df = pd.concat([volume_tracings_data_frame_test['Test'],
                    volume_tracings_data_frame_val['Validation'],
                    volume_tracings_data_frame_train['Train']],
                   axis=1)
    df.plot.density()
    plt.xlabel('Volume')
    plt.title('Distribution of Volume')
    plt.show()

    # get EF of all videos/echocardiograms
    metadata_path = Path(args.input_directory, args.metadata_filename)
    file_list_data_frame = pandas.read_csv(metadata_path)
    ef = file_list_data_frame[['EF', 'ESV', 'EDV']]

    # histogram of EF (bins contain EF's from 0-0.5, 0.5-1, ...)
    plt.hist(file_list_data_frame[['EF']], color='blue', edgecolor='black',
             bins=int(file_list_data_frame[['EF']].max()*2))
    plt.xlabel('EF')
    plt.ylabel('Number of Echocardiograms')
    plt.title('Histogram of EF')
    plt.show()

    # EF: density plot
    sns.displot(ef[['EDV']], kde=True)
    plt.xlabel('EF')
    plt.ylabel('Density')
    plt.title('Density plot of EF')
    plt.show()

    print('Mean', np.mean(ef[['ESV']]))
    print('Mean', np.mean(ef[['EDV']]))
    print('Std', np.std(ef[['ESV']]))
    print('Std', np.std(ef[['EDV']]))

    # boxplots of EF, ESV, EDV
    boxplot(ef[['EF']], 'EF', 'Boxplot of EF')
    boxplot(ef[['ESV']], 'ESV', 'Boxplot of ESV')
    boxplot(ef[['EDV']], 'EDV', 'Boxplot of EDV')

    # scatter plot of ESV and EDV
    ef.plot('ESV', 'EDV', kind='scatter', s=0.2)
    plt.xlabel('ESV')
    plt.ylabel('EDV')
    plt.title('Correlation of ESV and EDV')
    plt.show()


def boxplot(data, ylabel, title):
    plt.boxplot(data)
    plt.xticks([])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    main()
