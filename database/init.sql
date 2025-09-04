-- Diabetes Health Indicators Database Initialization Script
-- This script creates the initial database schema

-- Create the diabetes_health database if it doesn't exist
-- Note: This is handled by POSTGRES_DB environment variable in docker-compose

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create patients table
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    age INTEGER NOT NULL CHECK (age >= 0 AND age <= 150),
    sex INTEGER NOT NULL CHECK (sex IN (0, 1)),
    cp INTEGER NOT NULL CHECK (cp IN (0, 1, 2, 3)),
    trestbps INTEGER NOT NULL CHECK (trestbps > 0),
    chol INTEGER NOT NULL CHECK (chol > 0),
    fbs INTEGER NOT NULL CHECK (fbs IN (0, 1)),
    restecg INTEGER NOT NULL CHECK (restecg IN (0, 1, 2)),
    thalach INTEGER NOT NULL CHECK (thalach > 0),
    exang INTEGER NOT NULL CHECK (exang IN (0, 1)),
    oldpeak DECIMAL(3,1) NOT NULL CHECK (oldpeak >= 0),
    slope INTEGER NOT NULL CHECK (slope IN (0, 1, 2)),
    ca INTEGER NOT NULL CHECK (ca >= 0 AND ca <= 4),
    thal INTEGER NOT NULL CHECK (thal IN (0, 1, 2, 3))
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50) NOT NULL,
    probability DECIMAL(5,4) NOT NULL CHECK (probability >= 0 AND probability <= 1),
    prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
    risk_level VARCHAR(10) NOT NULL CHECK (risk_level IN ('Baixo', 'Médio', 'Alto'))
);

-- Create model_metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    roc_auc DECIMAL(5,4),
    training_samples INTEGER,
    test_samples INTEGER,
    features_count INTEGER
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_patients_created_at ON patients(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_patient_id ON predictions(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_model_metrics_created_at ON model_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_model_metrics_model_name ON model_metrics(model_name);

-- Insert sample data (optional, for testing)
INSERT INTO patients (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) 
VALUES 
    (63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1),
    (37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2),
    (41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2)
ON CONFLICT DO NOTHING;

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_patients_updated_at 
BEFORE UPDATE ON patients 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Log initialization completion
DO $$ 
BEGIN 
    RAISE NOTICE 'Diabetes Health Indicators database initialized successfully!';
END 
$$;