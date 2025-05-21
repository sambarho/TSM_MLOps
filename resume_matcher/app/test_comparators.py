import pytest
import comparators as comp

@pytest.mark.skip
def test_title_match_scores(resume_titles, job_title, bi_model):
    # Test with a single title
    resume_titles = ["Software Engineer", "Data Scientist"]
    job_title = "Software Engineer"
    bi_model = None  # Replace with actual model
    logger = None  # Replace with actual logger

    best_bi_pct, best_pair = comp.title_match_scores(resume_titles, job_title, bi_model, logger)

    assert isinstance(best_bi_pct, float)

    assert isinstance(best_pair, tuple)
    assert len(best_pair) == 2
    assert isinstance(best_pair[0], str)
    assert isinstance(best_pair[1], str)

@pytest.fixture
def bi_model():
    return { 'mock model' }
