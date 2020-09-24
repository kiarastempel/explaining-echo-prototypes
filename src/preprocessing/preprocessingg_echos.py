import os.path
from pathlib import Path
import cv2
import numpy as np
import pydicom as dicom


def mask_image(output):
    dimension = output.shape[0]

    # Mask pixels outside of scanning sector
    m1, m2 = np.meshgrid(np.arange(dimension), np.arange(dimension))

    mask = ((m1 + m2) > int(dimension / 2))
    mask *= ((m1 - m2) < int(dimension / 2))
    mask = np.reshape(mask, (dimension, dimension)).astype(np.int8)
    masked_image = cv2.bitwise_and(output, output, mask=mask)

    return masked_image


def make_video(file_to_process, destination_folder):
    file_name = file_to_process.stem
    crop_size = (768, 768)
    video_filename = (destination_folder / file_name).with_suffix('.avi')
    if not Path(video_filename).is_file():

        dataset = dicom.dcmread(file_to_process, force=True)
        test_array = dataset.pixel_array

        frames, height, width, channels = test_array.shape

        bias = int(np.abs(test_array.shape[2] - test_array.shape[1]) / 2)
        if bias > 0:
            if test_array.shape[1] < test_array.shape[2]:
                test_array = test_array[:, :, bias:-bias, :]
            else:
                test_array = test_array[:, bias:-bias, :, :]

        fps = 50

        try:
            fps = dataset[(0x18, 0x40)].value
        except KeyError:
            print(f"could not find frame rate, default to {fps}")

        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

        out = cv2.VideoWriter(str(video_filename), cv2.CAP_ANY, fourcc, fps, crop_size, isColor=False)

        for i in range(frames):
            output_a = test_array[i, :, :, 0]
            #output_a = cv2.cvtColor(output_a, cv2.COLOR_RGB2GRAY)
            small_output = output_a[int(0+height/5):(height - int(height / 20)),
                           int(height / 10):(height - int(height / 16))]

            # Resize image
            # Resize image
            output = cv2.resize(small_output, crop_size, interpolation=cv2.INTER_CUBIC)

            finaloutput = mask_image(output)

            out.write(finaloutput)

        out.release()

    else:
        print(file_name, "hasAlreadyBeenProcessed")

    return 0


def main():
    data_path = Path('../../data/raw')
    result_path = Path('../../data/processed')
    p = data_path.glob('**/*')
    video_paths = [x for x in p if x.is_file()]
    for video_path in video_paths:
        make_video(video_path, result_path)


if __name__ == "__main__":
    main()
