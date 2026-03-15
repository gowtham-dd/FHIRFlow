# fix_tickets_table.py
import sqlite3

DB_PATH = "data/claims.db"

conn = sqlite3.connect(DB_PATH)

# Check existing columns
cols = [row[1] for row in conn.execute("PRAGMA table_info(tickets)").fetchall()]
print("Existing columns:", cols)

# Add missing columns only if they don't exist
if "outcome" not in cols:
    conn.execute("ALTER TABLE tickets ADD COLUMN outcome TEXT")
    print("✅ Added: outcome")

if "open_email_sent" not in cols:
    conn.execute("ALTER TABLE tickets ADD COLUMN open_email_sent INTEGER DEFAULT 0")
    print("✅ Added: open_email_sent")

if "closed_email_sent" not in cols:
    conn.execute("ALTER TABLE tickets ADD COLUMN closed_email_sent INTEGER DEFAULT 0")
    print("✅ Added: closed_email_sent")

conn.commit()
conn.close()
print("\n✅ tickets table updated — restart your app now")