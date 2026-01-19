import sqlite3
from typing import Dict, Any, Optional, List

from vector_db.storage.schema import Schema


class SQLiteDB:
    def __init__(self, table_name: str, schema, db_path="data.db"):
        self.table_name = table_name
        self.schema = schema
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        # Build fields from schema
        fields = []
        for k, t in self.schema.all_fields().items():
            sql_type = self._map_type(t)
            fields.append(f"{k} {sql_type}")

        fields_sql = ", ".join(fields)

        cur = self.conn.cursor()
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            uid INTEGER PRIMARY KEY AUTOINCREMENT,
            {fields_sql},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        # Add indexes for faster time-based queries
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_created ON {self.table_name}(created_at)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_updated ON {self.table_name}(updated_at)")
        self.conn.commit()

    def _map_type(self, py_type):
        if py_type is str:
            return "TEXT"
        elif py_type is float:
            return "REAL"
        elif py_type is int:
            return "INTEGER"
        elif py_type is bytes:
            return "BLOB"
        else:
            return "TEXT"  # fallback

    # ---------- CRUD Methods ----------

    def add_item(self, data: Dict[str, Any]) -> int:
        """Insert a row into the table. Validates required fields."""
        self.schema.validate(data)
        keys = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        values = tuple(data.values())

        cur = self.conn.cursor()
        cur.execute(
            f"INSERT INTO {self.table_name} ({keys}) VALUES ({placeholders})",
            values
        )
        self.conn.commit()
        return cur.lastrowid

    def add_many_items(self, data_list: List[Dict[str, Any]]) -> List[int]:
        """
        Batch insert items, returns their UIDs.
        """
        if not data_list:
            return []

        cur = self.conn.cursor()

        # Build insert query
        fields = list(data_list[0].keys())
        placeholders = ", ".join(["?" for _ in fields])
        sql = f"INSERT INTO {self.table_name} ({', '.join(fields)}) VALUES ({placeholders})"

        # Insert all rows
        values = [tuple(item[field] for field in fields) for item in data_list]
        cur.executemany(sql, values)
        self.conn.commit()

        # Fetch last N IDs (sequential autoincrement)
        count = len(data_list)
        cur.execute(f"SELECT uid FROM {self.table_name} ORDER BY uid DESC LIMIT ?", (count,))
        rows = cur.fetchall()
        return [r[0] for r in reversed(rows)]

    def get_item(self, uid: int) -> Optional[Dict[str, Any]]:
        """Retrieve a row by UID, including automatic timestamps."""
        cur = self.conn.cursor()
        cur.execute(f"SELECT * FROM {self.table_name} WHERE uid=?", (uid,))
        row = cur.fetchone()
        if row is None:
            return None

        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, row))

    def get_items(self) -> List[Dict[str, Any]]:
        """
        Return all items from the table as a list of dicts.
        """
        cur = self.conn.cursor()
        cur.execute(f"SELECT * FROM {self.table_name}")
        rows = cur.fetchall()

        col_names = [desc[0] for desc in cur.description]
        return [dict(zip(col_names, row)) for row in rows]

    def get_items_by_ids(self, uids: list[int]) -> list[Dict[str, Any]]:
        """
        Retrieve multiple rows by their UIDs.
        Returns a list of dicts.
        """
        if not uids:
            return []
        placeholders = ",".join("?" for _ in uids)
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT * FROM {self.table_name} WHERE uid IN ({placeholders})",
            uids,
        )
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in rows]

    def get_all_items(self, limit: int = None) -> list[Dict[str, Any]]:
        """
        Retrieve all rows (optionally limited).
        """
        cur = self.conn.cursor()
        sql = f"SELECT * FROM {self.table_name}"
        if limit:
            sql += f" LIMIT {limit}"
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in rows]

    def update_item(self, uid: int, updates: Dict[str, Any]) -> bool:
        """Update given fields. Only schema fields are allowed. Always refreshes updated_at."""
        valid_fields = self.schema.all_fields().keys()
        updates = {k: v for k, v in updates.items() if k in valid_fields}
        if not updates:
            return False

        set_clause = ", ".join(f"{k}=?" for k in updates.keys())
        values = tuple(updates.values()) + (uid,)

        cur = self.conn.cursor()
        cur.execute(
            f"""
            UPDATE {self.table_name}
            SET {set_clause}, updated_at=CURRENT_TIMESTAMP
            WHERE uid=?
            """,
            values,
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_item(self, uid: int) -> bool:
        """Delete a row by UID."""
        cur = self.conn.cursor()
        cur.execute(f"DELETE FROM {self.table_name} WHERE uid=?", (uid,))
        self.conn.commit()
        return cur.rowcount > 0

    def delete_table(self) -> bool:
        """
        Drop the entire table from the database.
        WARNING: This deletes all data in the table.
        Returns True if successful.
        """
        cur = self.conn.cursor()
        try:
            cur.execute(f"DROP TABLE IF EXISTS {self.table_name}")
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting table {self.table_name}: {e}")
            return False

    def clear_table(self) -> bool:
        """
        Delete all rows from the table but keep schema.
        Equivalent to a reset.
        """
        cur = self.conn.cursor()
        try:
            cur.execute(f"DELETE FROM {self.table_name}")
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error clearing table {self.table_name}: {e}")
            return False