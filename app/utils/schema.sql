-- Initialize database schema for CyberImage

-- Drop existing tables if they exist
DROP TABLE IF EXISTS jobs;
DROP TABLE IF EXISTS images;

-- Create jobs table for queue management
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
    model_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    negative_prompt TEXT,
    settings TEXT NOT NULL,  -- JSON string of generation settings
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT
);

-- Create images table for generated images
CREATE TABLE images (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT NOT NULL,  -- JSON string of image metadata
    FOREIGN KEY (job_id) REFERENCES jobs (id)
);

-- Create indexes for common queries
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created_at ON jobs(created_at);
CREATE INDEX idx_images_job_id ON images(job_id);