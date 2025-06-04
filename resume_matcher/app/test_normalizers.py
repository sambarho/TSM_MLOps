import pytest
import normalizers

@pytest.mark.parametrize("job_title_original,job_title_normalized", [
    ('Senior Programmer', 'senior engineer'),
    ('Spaceship Cleaner in Command (120 %)', 'spaceship cleaner in command'),
    ('(Greenhorn) Apprentice Navigator', 'apprentice navigator'),
    ])
def test_normalize_title(job_title_original, job_title_normalized):
    # Act
    result = normalizers.normalize_title(job_title_original)
    # Assert
    assert result == job_title_normalized

@pytest.mark.parametrize("skill_original,skill_normalized", [
    (['ML', 'RNN', 'CNN'], ['ml', 'dl']),
    (['pytest', 'unit testing'], ['testing']),
    (['Scrum', 'Kanban'], ['agile']),
    ])
def test_normalize_skills(skill_original, skill_normalized):
    # Act
    result = normalizers.normalize_skills(skill_original)
    # Assert
    assert result == skill_normalized

@pytest.mark.skip
@pytest.mark.parametrize("skill_original,skill_normalized", [
    (['understands client needs', 'client focus'], ['customer-orientation']),
    ])
def test_normalize_soft_skills(skill_original, skill_normalized):
    # Act
    result = normalizers.normalize_soft_skills(skill_original, model)
    # Assert
    assert result == skill_normalized

@pytest.fixture
def model():
    return { 'mock model' }
