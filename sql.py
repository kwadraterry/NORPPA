import psycopg2
import numpy as np
from datetime import datetime


def create_connection():
    """ create a database connection to the POSTGRES database
    """
    conn = None
    try:
        conn = psycopg2.connect(
            # host="172.17.0.1",
            host="127.0.0.1",
            database="main",
            user="norppa",
            password="norppa")
    except psycopg2.Error as e:
        print(e)
    print("CONNECTION", conn)
    return conn


def create_tasks_table(conn):

    c = conn.cursor()

    # Create table - TASKS
    c.execute(
        """CREATE TABLE IF NOT EXISTS TASKS
                ([generated_id] INTEGER PRIMARY KEY,[task_name] text, [type] text, [params] text, [status_name] text, [status_val] integer, [begin_date] datetime)"""
    )
    c.close()
    conn.commit()

    return


def create_task(conn, task):

    sql = """ INSERT INTO TASKS(task_name, type, params, status_name, status_val, begin_date)
              VALUES(%s,%s,%s,%s,%s,%s) """
    cur = conn.cursor()
    cur.execute(sql, task)
    result = cur.lastrowid
    cur.close()
    return result


def get_task_status(conn, task_name):
    sql = """ SELECT status_name, status_val FROM TASKS WHERE task_name = %s"""
    cur = conn.cursor()
    cur.execute(sql, (task_name,))

    result = cur.fetchone()
    cur.close()
    return result


def get_task_by_id(conn, task_name):
    cur = conn.cursor()
    cur.execute(
        "SELECT task_name FROM TASKS WHERE generated_id=%s LIMIT 1", (task_name,)
    )

    result = cur.fetchone()
    cur.close()
    return result


def get_first_with_status(conn, status):
    sql = """ SELECT generated_id, task_name, type, params FROM TASKS WHERE status_name = %s ORDER BY begin_date LIMIT 1 ;"""
    cur = conn.cursor()
    cur.execute(sql, (status,))

    result = cur.fetchone()
    cur.close()
    return result


def update_status_val(conn, task_id, status_val):
    sql = """ UPDATE tasks SET status_val = %s WHERE generated_id = %s ;"""
    cur = conn.cursor()
    # cur.execute(sql, (status_val, task_id))
    cur.execute(sql, (status_val, task_id))
    conn.commit()
    return True


def update_status(conn, task_id, status):
    sql = """ UPDATE tasks SET status_name = %s WHERE generated_id = %s ;"""
    cur = conn.cursor()
    # cur.execute(sql, (status, task_id))
    cur.execute(sql, (status, task_id))
    conn.commit()
    return True


def create_database_table(conn):
    c = conn.cursor()

    c.execute("PRAGMA foreign_keys=ON")

    c.execute('''
            CREATE TABLE IF NOT EXISTS database
            ([image_id] INTEGER PRIMARY KEY AUTOINCREMENT, [seal_id] TEXT, [image_path] TEXT, [seal_type] TEXT,[encoding] TEXT, [date] TEXT)
            ''')
    c.close()
                        
    conn.commit()

def create_patches_table(conn):
    c = conn.cursor()

    c.execute('''
            CREATE TABLE IF NOT EXISTS patches
            ([patch_id] INTEGER PRIMARY KEY AUTOINCREMENT, [image_id] INTEGER, [coordinates] TEXT, [encoding] TEXT, FOREIGN KEY (image_id) REFERENCES database(image_id) ON DELETE CASCADE)
            ''')
            
    c.close() 
    conn.commit()

def create_users_table(conn):
    c = conn.cursor()

    c.execute('''
            CREATE TABLE IF NOT EXISTS users
            ([user_id] INTEGER PRIMARY KEY AUTOINCREMENT, [username] varchar(25) NOT NULL, [password] varchar(30) NOT NULL, UNIQUE (username))
            ''')
            
    c.close() 
    conn.commit()

def clean_database_table(conn, seal_type):
    c = conn.cursor()

    c.execute("DELETE FROM database WHERE seal_type = %s", (seal_type,))
                       
    conn.commit()

    c.close()

    return 


def delete_seal(conn, seal_type, seal_id):
    c = conn.cursor()

    c.execute("DELETE FROM database WHERE seal_type = %s AND  seal_id = %s", (seal_type, seal_id))
                       
    conn.commit()

    c.close()

    return 

def delete_image(conn, image_id):
    c = conn.cursor()

    c.execute("SELECT image_path FROM database WHERE image_id = %s", (image_id,))
    result = c.fetchone()
    c.execute("DELETE FROM database WHERE image_id = %s", (image_id,))
                       
    conn.commit()

    c.close()

    return result

def insert_database(conn, image_path, seal_id, seal_type, encoding, date):
    c = conn.cursor()

    c.execute("INSERT INTO database (image_path, seal_id, seal_type, encoding, date) VALUES (%s, %s, %s, %s, %s)", (image_path, seal_id, seal_type, np.array2string(encoding)[1:-1], date))
    row_id = c.lastrowid      
    conn.commit()

    c.close()
    return row_id

def insert_patches(conn, image_id, coordinates, encoding):
    c = conn.cursor()

    c.execute("INSERT INTO patches (image_id, coordinates, encoding) VALUES (%s, %s, %s)", (image_id, np.array2string(coordinates)[1:-1], np.array2string(encoding)[1:-1]))
                        
    conn.commit()

    c.close()

def update_encoding(conn, image_id, encoding, date):
    c = conn.cursor()

    sql = """ UPDATE database SET encoding = %s, date = %s WHERE image_id = %s """

    conn.execute(sql, (np.array2string(encoding)[1:-1], date, image_id))                    
    conn.commit()

    c.close()

def select_class_path_by_ids(conn, ids):
    c = conn.cursor()
    bindings = ",".join(["%s"] * len(ids))
    c.execute(f"SELECT image_path, seal_id FROM database WHERE image_id IN ({bindings})", [int(x) for x in ids])
    
    result = c.fetchall()
    c.close()
    return result


def get_encodings(conn, seal_type):
    c = conn.cursor()

    c.execute("SELECT image_id, encoding FROM database WHERE seal_type = %s ORDER BY image_id", (seal_type,))

    result = c.fetchall()
    c.close()
    image_ids = [res[0] for res in result]
    db_features = np.array([np.fromstring(res[1], dtype=float, sep=' ') for res in result])

    return image_ids, db_features

def get_db_ids(conn, seal_type):
    c = conn.cursor()

    c.execute("SELECT image_id FROM database WHERE seal_type = %s ORDER BY image_id", (seal_type,))

    result = c.fetchall()
    result = [res[0] for res in result]
    c.close()
    return result

def get_img_paths_by_id(conn, seal_id):
    c = conn.cursor()

    c.execute("SELECT image_id, image_path FROM database WHERE seal_id = %s", (seal_id,))

    result = c.fetchall()

    image_ids = [res[0] for res in result]
    image_paths = [res[1] for res in result]
    c.close()
    return image_ids, image_paths

def clean_patches(conn, img_id):
    c = conn.cursor()

    c.execute("DELETE FROM patches WHERE image_id = %s", (img_id, ))
                       
    conn.commit()

    c.close()

    return 


def get_patches(conn, image_id):
    c = conn.cursor()

    c.execute("SELECT coordinates, encoding FROM patches WHERE image_id = %s", (int(image_id),))

    result = c.fetchall()
    coordinates = np.array([np.fromstring(res[0], dtype=np.float32, sep=' ') for res in result])
    db_features = np.array([np.fromstring(res[1], dtype=np.float32, sep=' ') for res in result])
    c.close()
    return coordinates, db_features

def get_patch_features(conn, image_id):
    c = conn.cursor()

    c.execute("SELECT encoding FROM patches WHERE image_id = %s", (int(image_id),))

    result = c.fetchall()
    db_features = np.array([np.fromstring(res[0], dtype=np.float32, sep=' ') for res in result])
    c.close()
    return db_features

def get_patch_features_multiple_ids(conn, ids):
    c = conn.cursor()

    bindings = ",".join(["%s"] * len(ids))
    c.execute(f"SELECT image_id, encoding FROM patches WHERE image_id IN ({bindings})", [int(x) for x in ids])
    
    result = c.fetchall()

    db_features = np.array([np.fromstring(res[1], dtype=np.float32, sep=' ') for res in result])
    db_ids = np.array([int(res[0]) for res in result])
    c.close()
    return db_ids, db_features


def get_database_sql(conn, seal_type):
    c = conn.cursor()

    c.execute("SELECT image_id, image_path, seal_id, date FROM database WHERE seal_type = %s", (seal_type,))

    result = c.fetchall()
    c.close()
    return result


def get_tasks_sql(conn):
    c = conn.cursor()

    c.execute("SELECT * FROM tasks WHERE date(begin_date) BETWEEN current_date and current_date + interval '10 days';")

    result = c.fetchall()
    c.close()
    return result


def insert_user(conn, username, password):

    c = conn.cursor()
    try: 
        c.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        conn.commit()
    except Exception as e:
        return False, e         
    finally:
        c.close()

    return True, None


def get_password(conn, username):
    c = conn.cursor()

    c.execute("SELECT password FROM users WHERE username = %s", (username,))

    result = c.fetchone()
    c.close()
    return result


def set_task_to_failed():
    conn = create_connection()
    sql = """ UPDATE tasks SET status_name='failed' WHERE status_name not in ('sent', 'ready');"""
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    return True