import pyodbc


class CHandleDatabase:
    global _cursor

    def __init__(self):
        _cursor = self.connect_db()


    # open database connection
    def connect_db(self):
        server = 'GHSS-DBSERV\DEV'
        database = 'SvensSpielwiese'
        cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';Trusted_Connection=True')
        cursor = cnxn.cursor()
        return cursor

    def get_id(self, subdirname):
        global _cursor
        result = _cursor.execute('SELECT myoid FROM dbo.TableLoadStatus')