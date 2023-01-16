import sqlite3
import numpy as np
from datetime import datetime


def create_connection(path="/app/mount/tasks.db"):
    """ create a database connection to the SQLite database
    """
    conn = None
    try:
        conn = sqlite3.connect(path, check_same_thread=False)
    except sqlite3.Error as e:
        print(e)

    return conn



def select_class_path_by_ids(conn, ids):
    c = conn.cursor()
    bindings = ",".join(["?"] * len(ids))
    c.execute(f"SELECT image_path, seal_id FROM database WHERE image_id IN ({bindings})", [int(x) for x in ids])
    
    return c.fetchall()

def create_database_table(conn):
    c = conn.cursor()

    c.execute("PRAGMA foreign_keys=ON")

    c.execute('''
            CREATE TABLE IF NOT EXISTS database
            ([image_id] INTEGER PRIMARY KEY AUTOINCREMENT, [seal_id] TEXT, [image_path] TEXT, [seal_type] TEXT,[encoding] TEXT, [date] TEXT)
            ''')
    c.close()
                        
    conn.commit()

def get_patch_features(conn, image_id):
    c = conn.cursor()

    c.execute("SELECT encoding FROM patches WHERE image_id = ?", (int(image_id),))

    result = c.fetchall()
    db_features = np.array([np.fromstring(res[0], dtype=np.float32, sep=' ') for res in result])
    c.close()
    return db_features

def get_patch_features_multiple_ids(conn, ids):
    c = conn.cursor()

    bindings = ",".join(["?"] * len(ids))
    c.execute(f"SELECT image_id, encoding FROM patches WHERE image_id IN ({bindings})", [int(x) for x in ids])
    
    result = c.fetchall()

    db_features = np.array([np.fromstring(res[1], dtype=np.float32, sep=' ') for res in result])
    db_ids = np.array([int(res[0]) for res in result])
    c.close()
    return db_ids, db_features


def get_fisher_vectors(conn, ids):
    c = conn.cursor()
    bindings = ",".join(["?"] * len(ids))
    c.execute(f"SELECT encoding FROM database WHERE image_id IN ({bindings})", [int(x) for x in ids])
    
    result = c.fetchall()
    db_features = np.array([np.fromstring(res[0], dtype=float, sep=' ') for res in result])
    
    return db_features


def get_label(conn, image_id):
    c = conn.cursor()

    c.execute("SELECT seal_id FROM database WHERE image_id = ?", (image_id,))

    result = c.fetchone()

    return result

def get_db_ids(conn, seal_type="norppa"):
    c = conn.cursor()
    
    c.execute("SELECT image_id FROM database WHERE seal_type = ? ORDER BY image_id", (seal_type,))

    result = c.fetchall()
    result = [res[0] for res in result]
    return result

def get_img_paths_by_id(conn, seal_id):
    c = conn.cursor()

    c.execute("SELECT image_id, image_path FROM database WHERE seal_id = ?", (seal_id,))

    result = c.fetchall()

    image_ids = [res[0] for res in result]
    image_paths = [res[1] for res in result]

    return image_ids, image_paths




def get_patches(conn, image_id):
    c = conn.cursor()

    c.execute("SELECT coordinates, encoding FROM patches WHERE image_id = ?", (int(image_id),))

    result = c.fetchall()
    coordinates = np.array([np.fromstring(res[0], dtype=np.float32, sep=' ') for res in result])
    db_features = np.array([np.fromstring(res[1], dtype=np.float32, sep=' ') for res in result])

    return db_features, coordinates

def get_patches_multiple(conn, ids):
    c = conn.cursor()
    bindings = ",".join(["?"] * len(ids))
    c.execute(f"SELECT coordinates, encoding FROM patches WHERE image_id IN ({bindings})", [int(x) for x in ids])

    result = c.fetchall()
    coordinates = np.array([np.fromstring(res[0], dtype=np.float32, sep=' ') for res in result])
    db_features = np.array([np.fromstring(res[1], dtype=np.float32, sep=' ') for res in result])

    return db_features, coordinates

def get_patch_features(conn, image_id):
    c = conn.cursor()

    c.execute("SELECT encoding FROM patches WHERE image_id = ?", (int(image_id),))

    result = c.fetchall()
    db_features = np.array([np.fromstring(res[0], dtype=np.float32, sep=' ') for res in result])

    return db_features



def get_patch_features_multiple_ids_with_labels(conn, ids):
    c = conn.cursor()

    bindings = ",".join(["?"] * len(ids))
    c.execute(f"SELECT patches.image_id, patches.encoding, database.seal_id FROM patches INNER JOIN database ON patches.image_id = database.image_id WHERE patches.image_id IN ({bindings})", [int(x) for x in ids])
    
    result = c.fetchall()

    db_features = np.array([np.fromstring(res[1], dtype=np.float64, sep=' ') for res in result])
    db_ids = np.array([int(res[0]) for res in result])
    db_labels = np.array([res[2] for res in result])
    return db_ids, db_features, db_labels


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
    
    
def insert_patches(conn, image_id, coordinates, encoding):
    c = conn.cursor()

    c.execute("INSERT INTO patches (image_id, coordinates, encoding) VALUES (?, ?, ?)", (image_id, np.array2string(coordinates)[1:-1], np.array2string(encoding)[1:-1]))
                        
    conn.commit()

    c.close()

def update_encoding(conn, image_id, encoding, date):
    c = conn.cursor()

    sql = """ UPDATE database SET encoding = ?, date = ? WHERE image_id = ? """

    conn.execute(sql, (np.array2string(encoding)[1:-1], date, image_id))                    
    conn.commit()

    c.close()
    
    
def insert_database(conn, image_path, seal_id, seal_type, encoding, date):
    c = conn.cursor()

    c.execute("INSERT INTO database (image_path, seal_id, seal_type, encoding, date) VALUES (?, ?, ?, ?, ?)", (image_path, seal_id, seal_type, np.array2string(encoding)[1:-1], date))
    row_id = c.lastrowid      
    conn.commit()

    c.close()
    return row_id