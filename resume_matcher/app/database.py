'''
models.py

Defines the SQLAlchemy ORM model for storing resume-to-job matching records
and provides a utility function to save matching results to the database.
'''
from sqlalchemy import create_engine, Column, Integer, String, Text, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
import json

Base = declarative_base()

class MatchRecord(Base):
    '''
    ORM model representing a single record of a resume vs. job description match.

    Attributes:
        id (Integer): Primary key for the record.
        resume_name (String): Name of the resume file.
        resume_file (LargeBinary): Binary data of the resume file.
        job_description (Text): Raw job description text.
        resume_info (Text): JSON-serialized metadata extracted from the resume.
        job_info (Text): JSON-serialized metadata extracted from the job description.
        comparison_scores (Text): JSON-serialized similarity scores or other comparison metrics.
        timestamp (DateTime): UTC timestamp when the record was created (defaults to now).
    '''
    __tablename__ = "match_records"

    id = Column(Integer, primary_key=True)
    resume_name = Column(String)
    resume_file = Column(LargeBinary)
    job_description = Column(Text)
    resume_info = Column(Text)
    job_info = Column(Text)
    comparison_scores = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

def save_match_record(resume_file, job_description, resume_info, job_info, scores):
    '''
    Persist a new MatchRecord to the database.

    Reads the contents of the provided resume file, serializes resume/job metadata
    and comparison scores as JSON, and saves all information in a new database row.

    Parameters:
        resume_file (file-like): Open file object for the resume (must have a .name attribute).
        job_description (str): Raw text of the job description.
        resume_info (dict): Dictionary of metadata extracted from the resume.
        job_info (dict): Dictionary of metadata extracted from the job description.
        scores (dict): Dictionary containing similarity scores or other comparison metrics.

    Behavior:
        - Seeks to the beginning of the resume file to ensure full read.
        - Opens a new database session, creates a MatchRecord instance, and commits it.
        - Closes the session after the commit.
    '''
    resume_file.seek(0)  # Ensure we read from the beginning
    session = Session()
    record = MatchRecord(
        resume_name=resume_file.name,
        resume_file=resume_file.read(),
        job_description=job_description,
        resume_info=json.dumps(resume_info),
        job_info=json.dumps(job_info),
        comparison_scores=json.dumps(scores)
    )
    session.add(record)
    session.commit()
    session.close()

# SQLite DB inside container — ephemeral
engine = create_engine("sqlite:///resume_match.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
