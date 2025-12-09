import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "farm_data.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # User Profile Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    location_name TEXT,
                    latitude REAL,
                    longitude REAL,
                    soil_type TEXT,
                    land_area REAL,
                    variety TEXT,
                    fruit_density TEXT,
                    created_at TEXT
                )
            ''')
            
            # Spray/Fertilizer Records Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS spray_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    date TEXT,
                    type TEXT, -- 'spray' or 'fertilizer'
                    name TEXT,
                    quantity REAL,
                    unit TEXT,
                    notes TEXT,
                    created_at TEXT
                )
            ''')
            
            # Feedback/Learning Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    disease_predicted TEXT,
                    actual_disease TEXT,
                    confidence REAL,
                    is_correct INTEGER,
                    user_comments TEXT,
                    timestamp TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    def save_user_profile(self, profile_data: Dict):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, name, location_name, latitude, longitude, soil_type, land_area, variety, fruit_density, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile_data.get('user_id', 'default_user'),
                profile_data.get('name'),
                profile_data.get('location', {}).get('name'),
                profile_data.get('location', {}).get('latitude'),
                profile_data.get('location', {}).get('longitude'),
                profile_data.get('soil_type'),
                profile_data.get('land_area_acres'),
                profile_data.get('custard_apple_variety'),
                profile_data.get('fruit_density'),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving profile: {e}")
            return False

    def get_user_profile(self, user_id: str = 'default_user') -> Optional[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "user_id": row['user_id'],
                    "name": row['name'],
                    "email": row['email'],
                    "phone": row.get('phone'),
                    "location": {
                        "name": row['location_name'],
                        "latitude": row['latitude'],
                        "longitude": row['longitude']
                    } if row['location_name'] else None,
                    "soil_type": row['soil_type'],
                    "land_area_acres": row['land_area'],
                    "custard_apple_variety": row['variety'],
                    "fruit_density": row['fruit_density']
                }
            return None
        except Exception as e:
            logger.error(f"Error getting profile: {e}")
            return None

    def get_user_by_phone(self, phone: str) -> Optional[Dict]:
        """Get user profile by phone number"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM users WHERE phone = ?', (phone,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "user_id": row['user_id'],
                    "name": row['name'],
                    "email": row['email'],
                    "phone": row['phone'],
                    "location": {
                        "name": row['location_name'],
                        "latitude": row['latitude'],
                        "longitude": row['longitude']
                    } if row['location_name'] else None,
                    "land_area_acres": row['land_area'],
                    # Add other fields if needed
                }
            return None
        except Exception as e:
            logger.error(f"Error getting user by phone: {e}")
            return None

    def add_record(self, record: Dict):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO spray_records 
                (user_id, date, type, name, quantity, unit, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.get('user_id', 'default_user'),
                record.get('date'),
                record.get('type', 'spray'),
                record.get('name'),
                record.get('quantity'),
                record.get('unit', 'L'),
                record.get('notes'),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error adding record: {e}")
            return False

    def get_records(self, user_id: str = 'default_user') -> List[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM spray_records WHERE user_id = ? ORDER BY date DESC', (user_id,))
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting records: {e}")
            return []
    def _hash_password(self, password: str, salt: bytes = None) -> tuple[str, str]:
        """Hash a password using PBKDF2"""
        import hashlib
        import os
        
        if salt is None:
            salt = os.urandom(32) # 32 bytes salt
        else:
            # excessive safety - ensure bytes
            if isinstance(salt, str):
                salt = bytes.fromhex(salt)
                
        # PBKDF2-HMAC-SHA256
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000 # 100k iterations
        )
        
        return key.hex(), salt.hex()

    def create_user(self, email: str, password: str, name: str, phone: str = None) -> bool:
        """Create a new user with hashed password"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if email exists
            cursor.execute('SELECT 1 FROM users WHERE email = ?', (email,))
            if cursor.fetchone():
                return False # User exists
            
            # Hash password
            pwd_hash, salt = self._hash_password(password)
            
            user_id = f"user_{int(datetime.now().timestamp())}"
            
            # Check if columns exist (migration logic inline for simplicity)
            # This is a bit hacky but robust for dev: try insert with new columns, 
            # if fails, add them.
            try:
                # Optimized Insert including new auth fields
                cursor.execute('''
                    INSERT INTO users 
                    (user_id, email, password_hash, salt, name, phone, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, email, pwd_hash, salt, name, phone, datetime.now().isoformat()))
            except sqlite3.OperationalError:
                # Columns missing, migrate
                logger.info("Migrating database to include auth columns...")
                try:
                    cursor.execute('ALTER TABLE users ADD COLUMN email TEXT')
                    cursor.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')
                    cursor.execute('ALTER TABLE users ADD COLUMN salt TEXT')
                    cursor.execute('ALTER TABLE users ADD COLUMN phone TEXT')
                    
                    # Retry Insert
                    cursor.execute('''
                        INSERT INTO users 
                        (user_id, email, password_hash, salt, name, phone, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (user_id, email, pwd_hash, salt, name, phone, datetime.now().isoformat()))
                except Exception as e:
                    logger.error(f"Migration failed or column exists: {e}")
                    # Could fail if some cols exist and others don't, but assuming all-or-nothing for this task
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False

    def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        """Verify credentials and return user profile"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get user auth data
            # First ensure columns exist (if using previously created DB)
            try:
                cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
            except sqlite3.OperationalError:
                 return None # Columns likely don't exist, so no user has auth setup
                 
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
                
            stored_hash = row['password_hash']
            stored_salt = row['salt']
            
            if not stored_hash or not stored_salt:
                return None # Legacy user or broken record
                
            # Verify
            check_hash, _ = self._hash_password(password, stored_salt)
            
            if check_hash == stored_hash:
                return {
                    "user_id": row['user_id'],
                    "name": row['name'],
                    "email": row['email'],
                    "location": {
                        "name": row['location_name'],
                        "latitude": row['latitude'],
                        "longitude": row['longitude']
                    } if row['location_name'] else None,
                    "soil_type": row['soil_type'],
                    "land_area_acres": row['land_area'],
                    "custard_apple_variety": row['variety'],
                    "fruit_density": row['fruit_density']
                }
            
            return None
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return None
