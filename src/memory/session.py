import aiosqlite
from typing import Optional


class SessionMemory:
    def __init__(self, db_path: str = "./valura.db"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
    
    async def initialize(self):
        """Initialize database and create tables"""
        self._db = await aiosqlite.connect(self.db_path)
        
        # Create conversation_turns table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on session_id
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id 
            ON conversation_turns(session_id)
        """)
        
        await self._db.commit()
    
    async def add_turn(self, session_id: str, role: str, content: str):
        """Add a conversation turn to the database"""
        if self._db is None:
            await self.initialize()
        
        # Get max turn_index for this session
        cursor = await self._db.execute(
            "SELECT MAX(turn_index) FROM conversation_turns WHERE session_id = ?",
            (session_id,)
        )
        result = await cursor.fetchone()
        max_index = result[0] if result and result[0] is not None else -1
        next_index = max_index + 1
        
        # Insert new turn
        await self._db.execute(
            """INSERT INTO conversation_turns (session_id, turn_index, role, content)
               VALUES (?, ?, ?, ?)""",
            (session_id, next_index, role, content)
        )
        await self._db.commit()
    
    async def get_last_n_turns(self, session_id: str, n: int = 3) -> list[dict]:
        """Get last N turns for a session"""
        if self._db is None:
            await self.initialize()
        
        cursor = await self._db.execute(
            """SELECT role, content FROM conversation_turns 
               WHERE session_id = ? 
               ORDER BY turn_index DESC 
               LIMIT ?""",
            (session_id, n)
        )
        
        rows = await cursor.fetchall()
        # Reverse to get chronological order
        rows.reverse()
        
        return [{"role": row[0], "content": row[1]} for row in rows]
    
    async def clear_session(self, session_id: str) -> int:
        """Delete all turns for a session, return rowcount"""
        if self._db is None:
            await self.initialize()
        
        cursor = await self._db.execute(
            "DELETE FROM conversation_turns WHERE session_id = ?",
            (session_id,)
        )
        await self._db.commit()
        return cursor.rowcount
    
    async def close(self):
        """Close database connection"""
        if self._db:
            await self._db.close()
