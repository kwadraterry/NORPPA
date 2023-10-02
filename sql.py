import psycopg2
import numpy as np
from datetime import datetime
from pgvector.psycopg2 import register_vector

from reidentification.encoding_utils import aggregate_features
from tqdm import tqdm

def create_connection():
    """ create a database connection to the POSTGRES database
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="main",
            user="norppa",
            password="norppa")
        register_vector(conn)
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

def insert_database(conn, image_path, seal_id, date, viewpoints):
    c = conn.cursor()

    c.execute("INSERT INTO database (image_path, seal_id, date, viewpoint_right, viewpoint_left, viewpoint_up, viewpoint_down) VALUES (%s, %s, %s, %s, %s,%s, %s) RETURNING image_id", (image_path, seal_id,  date, bool(viewpoints["right"]), bool(viewpoints["left"]), bool(viewpoints["up"]), bool(viewpoints["down"])))
    row_id = c.fetchone()[0] 
    conn.commit()

    c.close()
    return row_id

def insert_patches(conn, image_id, coordinates, encoding):
    c = conn.cursor()
    register_vector(conn)

    c.execute("INSERT INTO patches (image_id, coordinates, encoding) VALUES (%s, %s, %s)", (image_id, coordinates, np.array(encoding)))
                        
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

    c.execute("SELECT patch_id, coordinates, encoding FROM patches WHERE image_id = %s", (int(image_id),))

    result = c.fetchall()
    ids = np.array([res[0] for res in result])
    coordinates = np.array([res[1] for res in result])
    db_features = np.array([res[2] for res in result])
    c.close()
    return ids, coordinates, db_features

def update_patch_coordinates(conn, patch_id, coordinates):
    c = conn.cursor()
    register_vector(conn)

    sql = """ UPDATE patches SET coordinates = %s WHERE patch_id = %s """
    
    c.execute(sql, (coordinates, int(patch_id)))                    
    conn.commit()

    c.close()

def get_patch_features(conn, image_id):
    c = conn.cursor()

    c.execute("SELECT encoding FROM patches WHERE image_id = %s", (int(image_id),))

    result = c.fetchall()
    db_features = np.array([np.fromstring(res[0], dtype=np.float32, sep=' ') for res in result])
    c.close()
    return db_features


def get_patch_coordinates(conn, image_id):
    c = conn.cursor()

    c.execute("SELECT patch_id, coordinates FROM patches WHERE image_id = %s", (int(image_id),))

    result = c.fetchall()
    # print(result)
    ids = np.array([res[0] for res in result])
    db_features = np.array([res[1] for res in result])
    c.close()
    return ids, db_features

def get_patches_multiple_ids(conn, ids):
    c = conn.cursor()

    bindings = ",".join(["%s"] * len(ids))
    c.execute(f"SELECT image_id, encoding, coordinates FROM patches WHERE image_id IN ({bindings})", [int(x) for x in ids])
    
    result = c.fetchall()

    db_features = np.array([res[1] for res in result]) 
    db_ellipses = np.array([res[2] for res in result]) 
    db_ids = np.array([int(res[0]) for res in result])
    c.close()
    return db_ids, db_features, db_ellipses

def get_all_fishers(conn, viewpoints, species):
    c = conn.cursor()
    encodings_str = ", ".join([f"encoding_{i}" for i in range(12)])
    viewpoints_str = ", ".join(["%s"] * len(viewpoints))
    
    c.execute("SELECT seal_id, viewpoint, {} FROM fisher_vectors WHERE viewpoint in ({}) AND seal_id IN (SELECT seal_id from seals WHERE species=%s)".format(encodings_str, viewpoints_str), 
              (*viewpoints, species))
    
    result = c.fetchall()

    # db_features = np.array([np.fromstring(res[2], dtype=np.float64, sep=' ') for res in result]) #(np.array2string(encoding)[1:-1]
    db_features = np.array([np.hstack(res[2:]) for res in result])
    db_ids = np.array([res[0] for res in result])
    db_viewpoints = np.array([res[1] for res in result])
    c.close()
    return db_ids, db_viewpoints, db_features

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def upload_fisher(conn, seal_id, viewpoint, encoding):
    cur = conn.cursor()
    encodings_str = ", ".join([f"encoding_{i}" for i in range(12)])
    format_str = ", ".join(["%s"] * 12)
    
    # cur.execute("INSERT INTO fisher_vectors (seal_id, viewpoint, encoding) VALUES (%s, %s, %s)", (seal_id, viewpoint, np.array2string(encoding, threshold=np.inf, max_line_width=np.inf)[1:-1]))
    cur.execute("INSERT INTO fisher_vectors (seal_id, viewpoint, {}) VALUES (%s, %s, {})".format(encodings_str, format_str), 
                (seal_id, viewpoint, *chunks(encoding, 16000)))
    
    
    conn.commit()
    return True

def get_database_sql(conn, species):
    c = conn.cursor()

    c.execute("SELECT database.image_id, database.image_path, database.seal_id, seals.seal_name, database.date, database.viewpoint_right, database.viewpoint_left, database.viewpoint_top, database.viewpoint_bottom FROM database INNER JOIN seals ON seals.seal_id=database.seal_id WHERE seals.species = %s", (species,))

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

def update_viewpoint_sql(conn, image_id, viewpoint):
    cur = conn.cursor()
    cur.execute("UPDATE database SET {0}= NOT {0} WHERE image_id = %s".format("viewpoint_" + viewpoint), (image_id,))
    conn.commit()
    return True


def set_task_to_failed():
    conn = create_connection()
    sql = """ UPDATE tasks SET status_name='failed' WHERE status_name not in ('sent', 'ready');"""
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    return True



def get_seal(conn, seal_id):
    c = conn.cursor()

    c.execute("SELECT seal_id, seal_name FROM seals WHERE seal_id = %s", (seal_id,))


    result = c.fetchone()
    c.close()
    
    return result

def update_seal_name(conn, seal_id, seal_name):
    cur = conn.cursor()
    cur.execute("UPDATE seals SET seal_name= %s WHERE seal_id = %s", (seal_name, seal_id))
    conn.commit()
    return True


def create_seal(conn, seal_id, seal_name, species):
    cur = conn.cursor()
    cur.execute("INSERT INTO seals (seal_id, seal_name, species) VALUES (%s, %s, %s)", (seal_id, seal_name, species))
    conn.commit()
    return True

def get_image(conn, seal_id, image_path):
    c = conn.cursor()

    c.execute("SELECT image_id FROM database WHERE seal_id = %s AND image_path= %s ", (seal_id,image_path))


    result = c.fetchone()
    c.close()
    
    if result is None:
        return None

    return result[0]

def get_images_by_seal_viewpoint(conn, seal_id, viewpoint):
    c = conn.cursor()

    c.execute("SELECT image_id FROM database WHERE seal_id = %s AND {0}=true".format("viewpoint_" + viewpoint), (seal_id,))

    result = c.fetchall()
    c.close()
    return result

def get_features_coordinates(conn, image_id):
    c = conn.cursor()

    c.execute("SELECT encoding,coordinates FROM patches where image_id=%s", (image_id,))

    result = c.fetchall()
    c.close()
    return result

def get_seals(conn, species):
    c = conn.cursor()

    c.execute("SELECT seal_id FROM seals WHERE species = %s", (species,))

    result = c.fetchall()
    c.close()
    return result

def get_features_for_aggregation(conn, seal_id, viewpoint):
    c = conn.cursor()

    c.execute("SELECT patches.encoding FROM patches INNER JOIN database ON patches.image_id=database.image_id WHERE database.seal_id = %s and database.{0}=true".format("viewpoint_" + viewpoint), (seal_id,))

    result = c.fetchall()
    c.close()
    return result

def aggregate_fisher_per_class(conn, codebooks, species, viewpoints = ["right", "left", "up", "down"]):
    all_seals = [x[0] for x in get_seals(conn, species)] # TODO
    db = []
    encoding_params = codebooks["gmm"]
    for seal_id in tqdm(all_seals):
        for viewpoint in viewpoints:
            all_features = get_features_for_aggregation(conn, seal_id, viewpoint)
            
            if len(all_features) == 0:
                continue
            all_features = np.array(all_features)[:, 0, :].astype(dtype=np.float64)
            encoded = aggregate_features(all_features, encoding_params) # from reidentification.encoding_utils
            db.append((encoded, {"class_id":seal_id, "viewpoint":viewpoint}))
    return db


def dataset_from_db(db_ids, db_viewpoints, db_features):
    return [(fisher, {'class_id':class_id, 'viewpoint':viewpoint}) for (class_id, viewpoint, fisher) in zip(db_ids, db_viewpoints, db_features)]

def dataset_from_sql(conn, viewpoints, species):
    return dataset_from_db(*get_all_fishers(conn, viewpoints, species))