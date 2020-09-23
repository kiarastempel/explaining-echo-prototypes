import os.path
from pathlib import Path
import cv2
import numpy as np
import pydicom as dicom


# from scipy.misc import imread
def create_mask(output):
    dimension = output.shape[0]

    # Mask pixels outside of scanning sector
    m1, m2 = np.meshgrid(np.arange(dimension), np.arange(dimension))

    mask = ((m1 + m2) > int(dimension / 2) + int(dimension / 10))
    mask *= ((m1 - m2) < int(dimension / 2) + int(dimension / 10))
    mask = np.reshape(mask, (dimension, dimension)).astype(np.int8)
    masked_image = cv2.bitwise_and(output, output, mask=mask)

    return masked_image


def make_video(file_to_process, destination_folder):
    file_name = file_to_process.split('\\')[-1]  # \\ if windows, / if on mac or sherlock
    crop_size = (1024, 1024)

    if not os.path.isdir(os.path.join(destination_folder, file_name)):

        dataset = dicom.dcmread(file_to_process, force=True)
        test_array = dataset.pixel_array

        frame_0 = test_array[0]
        mean = np.mean(frame_0, axis=1)
        mean = np.mean(mean, axis=1)
        y_crop = np.where(mean < 1)[0][0]
        test_array = test_array[:, y_crop:, :, :]

        bias = int(np.abs(test_array.shape[2] - test_array.shape[1]) / 2)
        if bias > 0:
            if test_array.shape[1] < test_array.shape[2]:
                test_array = test_array[:, :, bias:-bias, :]
            else:
                test_array = test_array[:, bias:-bias, :, :]

        print(test_array.shape)
        frames, height, width, channels = test_array.shape

        fps = 50

        try:
            fps = dataset[(0x18, 0x40)].value
        except ValueError:
            print("couldn't find frame rate, default to 50")

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_filename = os.path.join(destination_folder, file_name + '.avi')
        out = cv2.VideoWriter(video_filename, fourcc, fps, crop_size)

        for i in range(frames):
            output_a = test_array[i, :, :, 0]
            small_output = output_a[int(height / 14):(height - int(height / 14)),
                           int(height / 14):(height - int(height / 14))]

            # Resize image
            output = cv2.resize(small_output, crop_size, interpolation=cv2.INTER_CUBIC)

            final_output = create_mask(output)

            final_output = cv2.merge([final_output, final_output, final_output])
            out.write(final_output)

        out.release()

    else:
        print(file_name, "hasAlreadyBeenProcessed")

    return 0


def main():
    data_path = Path('../data/raw')
    result_path = Path('../data/processed')

    for video_path in videos:
        make_video(video_path, result_path)


if __name__ == "__main__":
    main()
