# migrate_db.py
import sqlite3
from create_db import DB_NAME

def migrate_camera_status():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        # Add status column to cameras table if it doesn't exist
        cursor.execute("PRAGMA table_info(cameras)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'status' not in columns:
            cursor.execute("ALTER TABLE cameras ADD COLUMN status TEXT DEFAULT 'stopped'")
            print("✅ Added status column to cameras table")
        
        # Update existing cameras to have 'stopped' status
        cursor.execute("UPDATE cameras SET status='stopped' WHERE status IS NULL")
        
        conn.commit()
        print("✅ Database migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_camera_status()