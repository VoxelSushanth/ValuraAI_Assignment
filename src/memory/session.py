import aiosqlite
import logging

logger = logging.getLogger(__name__)


class SessionMemory:
    """
    Async SQLite session memory.
    Uses per-operation connections via async context managers for reliability.
    Zero shared connection state — safe for concurrent async use.
    """

    def __init__(self, db_path: str = "./valura.db"):
        self.db_path = db_path

    async def initialize(self) -> None:
        """Create tables and indexes if they do not exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id
                ON conversation_turns(session_id)
            """)
            await db.commit()
        logger.info(f"SessionMemory initialized at {self.db_path}")

    async def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Add a conversation turn. Auto-increments turn_index per session."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT COALESCE(MAX(turn_index), -1) FROM conversation_turns WHERE session_id = ?",
                (session_id,)
            )
            row = await cursor.fetchone()
            next_index = (row[0] if row and row[0] is not None else -1) + 1

            await db.execute(
                """INSERT INTO conversation_turns (session_id, turn_index, role, content)
                   VALUES (?, ?, ?, ?)""",
                (session_id, next_index, role, content)
            )
            await db.commit()

    async def get_last_n_turns(self, session_id: str, n: int = 3) -> list[dict]:
        """Return last N turns for a session in chronological order."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT role, content FROM conversation_turns
                   WHERE session_id = ?
                   ORDER BY turn_index DESC
                   LIMIT ?""",
                (session_id, n)
            )
            rows = await cursor.fetchall()

        rows = list(reversed(rows))
        return [{"role": row[0], "content": row[1]} for row in rows]

    async def clear_session(self, session_id: str) -> int:
        """Delete all turns for a session. Returns number of rows deleted."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM conversation_turns WHERE session_id = ?",
                (session_id,)
            )
            await db.commit()
            return cursor.rowcount

    async def close(self) -> None:
        """No-op: per-operation connections close automatically. Kept for API compatibility."""
        pass
