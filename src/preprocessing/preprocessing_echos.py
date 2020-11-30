import argparse
import io
from pathlib import Path
import cv2
import numpy as np
import pydicom as dicom
from PIL import Image
from pydicom.encaps import encapsulate
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_path', default='../../data/raw', help="Directory with the echocardiograms.")
    parser.add_argument('-o', '--output_directory', default='../../data/processed', help="Directory to save "
                                                                                         "the processed echocardiograms "
                                                                                         "into.")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_directory = Path(args.output_directory)
    p = data_path.glob('**/*')
    video_paths = [x for x in p if x.is_file()]
    for video_path in tqdm(video_paths):
        make_video(video_path, output_directory)


def make_video(file_to_process, destination_folder):
    file_name = file_to_process.stem
    print(f'Processing {file_name}')
    video_filename = (destination_folder / file_name).with_suffix('.avi')
    dicom_filename = (destination_folder / file_name).with_suffix('.dcm')

    if not (Path(video_filename).is_file() and Path(dicom_filename).is_file()):
        dicom_file = dicom.dcmread(file_to_process)
        pixel_array = dicom_file.pixel_array
        frames, height, width, channels = pixel_array.shape

        video = []
        left_crop = int(width * 0.23)
        right_crop = int(width * 0.16)
        upper_crop = int(height * 0.17)
        lower_crop = int(height * 0.1)
        crop_size_dcm = (height - upper_crop - lower_crop,
                         width - left_crop - right_crop)
        crop_size_video = crop_size_dcm[::-1]
        fps = 50
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(str(video_filename), fourcc, fps, crop_size_video, isColor=False)
        for i in range(frames):
            # The first channel of a YCbCr image is the grey value
            output_a = pixel_array[i, :, :, 0]

            # Cropping the image
            small_output = output_a[upper_crop:(height - lower_crop), left_crop:width - right_crop]

            masked_output = mask_image(small_output)

            # Resize image
            # resized_output = cv2.resize(small_output, crop_size, interpolation=cv2.INTER_CUBIC)

            out.write(masked_output)

            output_bytes = io.BytesIO()
            img = Image.fromarray(masked_output)
            img.save(output_bytes, format('JPEG'))
            video.append(output_bytes.getvalue())

        out.release()

        # Create the dcm file
        dicom_file.PixelData = encapsulate(video)
        dicom_file.SamplesPerPixel = 1
        dicom_file.ImageType = 'DERIVED'
        dicom_file.PhotometricInterpretation = "MONOCHROME2"
        dicom_file.Rows, dicom_file.Columns = crop_size_dcm
        dicom_file.CineRate = 50
        dicom_file.save_as(dicom_filename, write_like_original=False)

    else:
        print(file_name, "hasAlreadyBeenProcessed")


def mask_image(output):
    y, x = output.shape

    # Mask pixels outside of scanning sector
    mask = np.ones((y, x), np.uint8) * 255

    left_triangle_left_corner = (0, 0)
    left_triangle_right_corner = (int(x / 2) - 4, 0)
    left_triangle_lower_corner = (0, 400)

    right_triangle_left_corner = (int(x / 2) - 4, 0)
    right_triangle_right_corner = (x, 0)
    right_triangle_lower_corner = (x, 400)

    triangle_left = np.array([left_triangle_left_corner, left_triangle_lower_corner, left_triangle_right_corner])
    triangle_right = np.array([right_triangle_right_corner, right_triangle_left_corner, right_triangle_lower_corner])
    cv2.drawContours(mask, [triangle_right, triangle_left], -1, 0, -1)
    return cv2.bitwise_and(output, output, mask=mask)


if __name__ == "__main__":
    main()
