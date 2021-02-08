import io
import os
from pathlib import Path
import cv2
import numpy as np
import pydicom as dicom
from PIL import Image
from pydicom.encaps import encapsulate
from tqdm import tqdm
import pyodbc
import shutil
import sqlserverport


#sudo docker run -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=ThatIsAPassword1234" -p 1433:1433 --name sql1 -h sql1 -d mcr.microsoft.com/mssql/server:2019-latest

class CHandleDatabase:


    def __init__(self):
        self._cursor = self.connect_db()


    # open database connection
    def connect_db(self):
        server = '10.8.4.205'
        database = 'DeepLearning'
        #port =  str(sqlserverport.lookup(server, ''))
        port = "1433"

        #self.cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';User=yes', timeout=180)
        self.cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';PORT='+port+';DATABASE=' + database + ';UID=sa;PWD=Db630475$', timeout=30)
        cursor = self.cnxn.cursor()
        print("SQL DB conncted")
        return cursor

    def get_campusID(self, subdirname):
        self._cursor.execute('SELECT myoid FROM dbo.TableEchoMeasureStatus WHERE REPLACE(studyuid, \'.\',\'\') like \'%' + subdirname + '%\' and status=2')
        row = self._cursor.fetchone()
        if(row!=None):
            myoid = row.myoid.strip()
            self._cursor.execute('SELECT id FROM dbo.UUIDtoMYOID WHERE myoid=\''+myoid+'\' and done_converting=0')
            uidrow = self._cursor.fetchone()
            if(uidrow!=None):
                return str(uidrow.id)
            else:
                #print("No CampusID found for: " + myoid)
                return None
        return None


    def get_campusID(self, subdirname):
        self._cursor.execute('SELECT myoid FROM dbo.TableEchoMeasureStatus where REPLACE(studyuid, \'.\',\'\') like \'%' + subdirname + '%\' and status=2')
        row = self._cursor.fetchone()
        if(row!=None):
            myoid = row.myoid.strip()
            self._cursor.execute('SELECT id, done_converting FROM dbo.UUIDtoMYOID WHERE myoid=\''+myoid+'\' and done_converting=0')
            uidrow = self._cursor.fetchone()
            if(uidrow!=None):
                return str(uidrow.id), myoid, uidrow.done_converting
            else:
                #print("No CampusID found for: " + myoid)
                return None, myoid, None
        return None, None, None

    def setCampusIDDone(self, campusid):
        self._cursor.execute("UPDATE dbo.UUIDtoMYOID SET done_converting=1 WHERE id= ?",  campusid )
        self.cnxn.commit()
        return True

    def getNextPathList(self, campusid):
        self._cursor.execute("UPDATE dbo.UUIDtoMYOID SET done_converting=1 WHERE id= ?",  campusid )
        self.cnxn.commit()
        return True

    # Identify view according to results
    def getView(self, myoid, filename):
        self._cursor.execute('SELECT * FROM dbo.TableResultsConventional WHERE myoid = \'' + myoid + '\' and filename=\''+filename+'.dcm\'')
        row = self._cursor.fetchone()
        if(row!=None):
            return row.view
        else:
            return None

def main():
    ## Initialise DB class
    DBHandler = CHandleDatabase()


    print("Curent dir:" + os.getcwd())
    #
    #result_path = Path(os.getcwd() +'/data/processed')
    #data_path = Path(os.getcwd() + '/data/raw')


    #result_path = Path('D:\processed')
    #data_path = Path('D:\Jul2020')


    result_path = Path('/home/tro2s/echoconverter/processedBL')
    data_path = Path('/mnt/windows_share/dicom_storage')

    p = data_path.glob('**/*')

    # Get all Subfolders of data_path
    subfolders = [e for e in data_path.iterdir() if e.is_dir()]
    for folder in subfolders:
        print(folder)

    #iterate through all paths
    for echofolder in tqdm(subfolders):
        video_paths = [x for x in echofolder.iterdir() if x.is_file()]
        # Identify subdir
        subdir = echofolder.parts[echofolder.parts.__len__()-1]

        #get corresponding CampusID
        campusid, myoid, done_converting = DBHandler.get_campusID(subdir)

        #check if ID was already processed. Beake if true
        #if(done_converting==1):
        #    break;

        counter = 0
        if(campusid!=None):
            for video_path in video_paths:
                try:
                    #set destination folder based on campusid
                    destination_folder = (result_path)
                    destination_folder.mkdir(parents=True, exist_ok=True)

                    #get view as classified
                    view = DBHandler.getView(myoid,video_path.stem)
                    if(view!=None):
                        #view = "unknown"
                        counter= counter + 1
                        make_video(video_path, destination_folder, campusid, view )


                except:
                    print("Error: " + str(video_path))

        ## If MYOID done set done
        if(counter<video_paths.__len__()):
            #shutil.rmtree(echofolder)
            DBHandler.setCampusIDDone(campusid)


def make_video(file_to_process, destination_folder, campusid, view):
    file_name = campusid + "_" + view + "_" + file_to_process.stem

    print(f'Processing {file_name}')

    video_filename = (destination_folder / file_name).with_suffix('.avi')
    dicom_filename = (destination_folder / file_name).with_suffix('.dcm')

    # Create Subdir


    if not (Path(video_filename).is_file() and Path(dicom_filename).is_file()):
        dicom_file = dicom.dcmread(file_to_process)
        pixel_array = dicom_file.pixel_array

        if(len(pixel_array.shape)==4):
            frames, height, width, channels = pixel_array.shape
        else:
            frames, height, width = pixel_array.shape

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
            if(len(pixel_array.shape)==4):
                # The first channel of a YCbCr image is the grey value
                output_a = pixel_array[i, :, :, 0]
            else:
                output_a = pixel_array[i, :, :]

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
        dicom_file.PatientID = campusid
        dicom_file.PatientName = campusid
        dicom_file.CineRate = 50
        #dicom_file.save_as(dicom_filename, write_like_original=False)

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