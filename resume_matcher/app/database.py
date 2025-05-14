from sqlalchemy import create_engine, Column, Integer, String, Text, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
import json

Base = declarative_base()

class MatchRecord(Base):
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
