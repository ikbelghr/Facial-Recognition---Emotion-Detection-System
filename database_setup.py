import sqlite3

def setup_database():
    conn = sqlite3.connect('customer_visits.db')
    c = conn.cursor()

    # Create tables for known faces, face images, and visits
    c.execute('''
        CREATE TABLE IF NOT EXISTS known_faces (
            customer_id TEXT PRIMARY KEY,
            face_encoding BLOB
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS face_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            image BLOB,
            timestamp TEXT,
            emotion TEXT,
            FOREIGN KEY (customer_id) REFERENCES known_faces (customer_id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            visit_timestamp TEXT,
            visit_count INTEGER DEFAULT 1,
            FOREIGN KEY (customer_id) REFERENCES known_faces (customer_id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS visit_emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            timestamp TEXT,
            dominant_emotion TEXT,
            FOREIGN KEY (customer_id) REFERENCES known_faces (customer_id)
        )
    ''')

    conn.commit()
    conn.close()

if __name__ == "__main__":
    setup_database()
