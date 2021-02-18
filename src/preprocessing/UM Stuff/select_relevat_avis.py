import io
import os
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
        server = 'localhost'
        database = 'DeepLearning'
        port = "1433"

        self.cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';PORT='+port+';DATABASE=' + database + ';UID=sa;PWD=ThatIsAPassword2020', timeout=30)
        cursor = self.cnxn.cursor()
        print("SQL DB conncted")
        return cursor

    def getBesta4c(self):
        self._cursor.execute('SELECT uuid, filename FROM dbo.besta4c')
        return  self._cursor.fetchall()

 

def main():
    
    data_path = "/mnt/c/Users/tro0s/Desktop/processedBL/done/"
    result_path = "/mnt/c/Users/tro0s/Desktop/PreprocessedBLFinal/"

    ## Initialise DB class
    DBHandler = CHandleDatabase()

    # Get besta4c
    files = DBHandler.getBesta4c()

    for row in tqdm(files):
        fullname =  row[0] + '_a4c_' +row[1]
        if os.path.isfile(result_path+fullname)==False:
        #os.system('copy ' + data_path + fullname + " "+result_path+fullname)
            shutil.copyfile(data_path + fullname, result_path+fullname)







if __name__ == "__main__":
    main()
