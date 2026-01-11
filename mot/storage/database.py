import sqlite3

def init_db(path):
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS tracklet (
        global_id INTEGER PRIMARY KEY,
        camera_id INTEGER
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS frame (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        global_id INTEGER,
        frame_id INTEGER,
        x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
        confidence REAL
    )""")

    conn.commit()
    return conn

