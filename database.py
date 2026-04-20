import sqlite3

def get_db():
    return sqlite3.connect("database/db.sqlite3")

def create_tables():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        symptoms TEXT,
        disease TEXT
    )
    """)

    conn.commit()
    conn.close()


def create_user(username, password):
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, password))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()


def authenticate_user(username, password):
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, password))

    user = c.fetchone()
    conn.close()
    return user


def save_history(user_id, symptoms, disease):
    conn = get_db()
    c = conn.cursor()

    c.execute(
        "INSERT INTO history (user_id, symptoms, disease) VALUES (?, ?, ?)",
        (user_id, symptoms, disease)
    )

    conn.commit()
    conn.close()


def get_history(user_id):
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT * FROM history WHERE user_id=?", (user_id,))
    data = c.fetchall()

    conn.close()
    return data