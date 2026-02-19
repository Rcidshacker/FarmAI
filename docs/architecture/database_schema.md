# Database Schema Documentation

## Database Details

- **Type:** SQLite
- **File Location:** `data/farm_data.db`
- **Management:** Raw SQL via `src/database/db_manager.py`

## Tables

### 1. `users`

**Purpose:** Stores user profiles and authentication credentials.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `user_id` | TEXT | PRIMARY KEY | Unique user identifier (e.g., "user_170736...") |
| `name` | TEXT | | Farmer's name |
| `email` | TEXT | | Email address (Added via migration) |
| `phone` | TEXT | | Phone number (Added via migration) |
| `password_hash` | TEXT | | PBKDF2 hashed password |
| `salt` | TEXT | | Salt for password hashing |
| `location_name` | TEXT | | e.g., "Pune" |
| `latitude` | REAL | | GPS Latitude |
| `longitude` | REAL | | GPS Longitude |
| `soil_type` | TEXT | | e.g., "Black Cotton Soil" |
| `land_area` | REAL | | Farm size in acres |
| `variety` | TEXT | | Custard Apple variety (e.g., "Phule Purandar") |
| `fruit_density` | TEXT | | Low/Medium/High |
| `created_at` | TEXT | | ISO8601 Timestamp |

### 2. `spray_records`

**Purpose:** Logs all farming interventions (spraying, fertilizing).

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique record ID |
| `user_id` | TEXT | | Foreign Key to `users.user_id` |
| `date` | TEXT | | Date of intervention |
| `type` | TEXT | | "spray" or "fertilizer" |
| `name` | TEXT | | Chemical/Fertilizer name |
| `quantity` | REAL | | Amount applied |
| `unit` | TEXT | | Unit (L, kg, etc.) |
| `notes` | TEXT | | User notes |
| `created_at` | TEXT | | Timestamp of record creation |

### 3. `feedback`

**Purpose:** Stores user feedback on disease detection accuracy (Active Learning).

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique feedback ID |
| `user_id` | TEXT | | Foreign Key to `users.user_id` |
| `disease_predicted` | TEXT | | Model prediction |
| `actual_disease` | TEXT | | User correction |
| `confidence` | REAL | | Model confidence score |
| `is_correct` | INTEGER | | 1 (True) or 0 (False) |
| `user_comments` | TEXT | | Additional context |
| `timestamp` | TEXT | | Time of feedback |

## Relationships

- **One-to-Many:** `users` (1) ↔ `spray_records` (Many) via `user_id`
- **One-to-Many:** `users` (1) ↔ `feedback` (Many) via `user_id`

## Migration Logic

The `db_manager.py` contains inline migration logic in `create_user()`. If insertion fails due to missing columns (OperationalError), it attempts to ALTER the `users` table to add:

- `email`
- `password_hash`
- `salt`
- `phone`
