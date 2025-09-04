-- Diabetes Health Indicators ML Database Schema
-- PostgreSQL DDL for project persistence

-- Create database diabetes_health automatically
CREATE DATABASE diabetes_health;

-- Diabetes raw data table
CREATE TABLE IF NOT EXISTS diabetes_raw (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Target variable
    diabetes_binary INT CHECK (diabetes_binary IN (0, 1)),
    
    -- Numeric features
    bmi DOUBLE PRECISION NOT NULL CHECK (bmi >= 10.0 AND bmi <= 100.0),
    ment_hlth DOUBLE PRECISION NOT NULL CHECK (ment_hlth >= 0 AND ment_hlth <= 30),
    phys_hlth DOUBLE PRECISION NOT NULL CHECK (phys_hlth >= 0 AND phys_hlth <= 30),
    
    -- Ordinal categorical features
    gen_hlth INT NOT NULL CHECK (gen_hlth >= 1 AND gen_hlth <= 5),
    age INT NOT NULL CHECK (age >= 1 AND age <= 13),
    education INT NOT NULL CHECK (education >= 1 AND education <= 6),
    income INT NOT NULL CHECK (income >= 1 AND income <= 8),
    
    -- Binary features
    high_bp INT NOT NULL CHECK (high_bp IN (0, 1)),
    high_chol INT NOT NULL CHECK (high_chol IN (0, 1)),
    chol_check INT NOT NULL CHECK (chol_check IN (0, 1)),
    smoker INT NOT NULL CHECK (smoker IN (0, 1)),
    stroke INT NOT NULL CHECK (stroke IN (0, 1)),
    heart_disease_or_attack INT NOT NULL CHECK (heart_disease_or_attack IN (0, 1)),
    phys_activity INT NOT NULL CHECK (phys_activity IN (0, 1)),
    fruits INT NOT NULL CHECK (fruits IN (0, 1)),
    veggies INT NOT NULL CHECK (veggies IN (0, 1)),
    hvy_alcohol_consump INT NOT NULL CHECK (hvy_alcohol_consump IN (0, 1)),
    any_healthcare INT NOT NULL CHECK (any_healthcare IN (0, 1)),
    no_docbc_cost INT NOT NULL CHECK (no_docbc_cost IN (0, 1)),
    diff_walk INT NOT NULL CHECK (diff_walk IN (0, 1)),
    sex INT NOT NULL CHECK (sex IN (0, 1)),
    asthma INT DEFAULT 0 CHECK (asthma IN (0, 1)),
    kidney_disease INT DEFAULT 0 CHECK (kidney_disease IN (0, 1)),
    skin_cancer INT DEFAULT 0 CHECK (skin_cancer IN (0, 1)),
    diabetic INT DEFAULT 0 CHECK (diabetic >= 0 AND diabetic <= 4)
);

-- Models metadata table (matches SQLAlchemy ORM ModelRun class)
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    auc DOUBLE PRECISION,
    f1 DOUBLE PRECISION,
    recall DOUBLE PRECISION,
    precision DOUBLE PRECISION,
    notes TEXT
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_diabetes_raw_target ON diabetes_raw(diabetes_binary);
CREATE INDEX IF NOT EXISTS idx_diabetes_raw_created_at ON diabetes_raw(created_at);
CREATE INDEX IF NOT EXISTS idx_diabetes_raw_bmi ON diabetes_raw(bmi);
CREATE INDEX IF NOT EXISTS idx_diabetes_raw_age ON diabetes_raw(age);

CREATE INDEX IF NOT EXISTS idx_model_runs_created_at ON model_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_model_runs_model_type ON model_runs(model_type);
CREATE INDEX IF NOT EXISTS idx_model_runs_test_score ON model_runs(test_score);

-- Insert sample data for testing (optional)
INSERT INTO diabetes_raw (
    diabetes_binary, bmi, ment_hlth, phys_hlth, gen_hlth, age, education, income,
    high_bp, high_chol, chol_check, smoker, stroke, heart_disease_or_attack,
    phys_activity, fruits, veggies, hvy_alcohol_consump, any_healthcare,
    no_docbc_cost, diff_walking, sex
) VALUES 
    (0, 25.0, 0, 0, 2, 5, 4, 6, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1),
    (1, 35.5, 5, 10, 4, 8, 3, 4, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0),
    (0, 22.8, 0, 2, 1, 3, 6, 8, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1)
ON CONFLICT DO NOTHING;

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_diabetes_raw_updated_at 
    BEFORE UPDATE ON diabetes_raw 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();