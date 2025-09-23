-- Initial database setup for TTS API
-- This file will be executed when the PostgreSQL container starts

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create initial user if using PostgreSQL
-- Note: This is just an example, in production you should create users through your application
-- INSERT INTO users (id, username, email, created_at)
-- VALUES (uuid_generate_v4(), 'admin', 'admin@example.com', NOW())
-- ON CONFLICT (username) DO NOTHING;