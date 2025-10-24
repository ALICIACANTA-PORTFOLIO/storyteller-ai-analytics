# 🤝 Contributing to AudioMind

First off, thank you for considering contributing to AudioMind! It's people like you that make AudioMind such a great tool.

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

---

## 📜 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Examples of behavior that contributes to creating a positive environment include:**

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Examples of unacceptable behavior include:**

- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at [your.email@example.com]. All complaints will be reviewed and investigated promptly and fairly.

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed
- **Git** for version control
- **Docker** (optional, for containerized development)
- **PostgreSQL 14+** and **Redis 7+** (if not using Docker)

### Setting Up Development Environment

1. **Fork the repository**

   Click the "Fork" button at the top right of the repository page.

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR_USERNAME/audiomind.git
   cd audiomind
   ```

3. **Add upstream remote**

   ```bash
   git remote add upstream https://github.com/original-owner/audiomind.git
   ```

4. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install dependencies**

   ```bash
   # Install with development dependencies
   pip install -r requirements/dev.txt
   
   # Install pre-commit hooks
   pre-commit install
   ```

6. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your settings (API keys, etc.)
   ```

7. **Set up database**

   ```bash
   # If using Docker
   docker-compose up -d postgres redis
   
   # Or install PostgreSQL and Redis locally
   # Then run migrations
   python scripts/setup_db.py
   alembic upgrade head
   ```

8. **Verify setup**

   ```bash
   # Run tests to ensure everything works
   pytest tests/unit/
   ```

---

## 🎯 How Can I Contribute?

### Reporting Bugs

**Before submitting a bug report:**

- Check the [existing issues](https://github.com/yourname/audiomind/issues) to avoid duplicates
- Collect information about the bug:
  - Stack trace
  - OS, Python version
  - Steps to reproduce
  - Expected vs actual behavior

**How to submit a good bug report:**

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Upload audio file '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.10.5]
- AudioMind version: [e.g., 0.2.1]

**Additional context**
Any other context about the problem.
```

### Suggesting Features

**Before submitting a feature request:**

- Check if the feature already exists
- Check the [roadmap](README.md#roadmap)
- Ensure your idea fits the project scope

**How to submit a good feature request:**

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md):

```markdown
**Is your feature request related to a problem?**
A clear description of the problem. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
Other solutions or features you've considered.

**Use case**
Who would benefit and how?

**Additional context**
Mockups, examples from other tools, etc.
```

### Contributing Code

We welcome code contributions! Areas where you can help:

- 🐛 **Bug fixes**: Issues labeled [`good first issue`](https://github.com/yourname/audiomind/labels/good%20first%20issue)
- ✨ **New features**: Issues labeled [`enhancement`](https://github.com/yourname/audiomind/labels/enhancement)
- 📚 **Documentation**: Issues labeled [`documentation`](https://github.com/yourname/audiomind/labels/documentation)
- 🧪 **Tests**: Improve test coverage
- 🎨 **UI/UX**: Improve dashboard design
- ⚡ **Performance**: Optimize code

### Contributing Documentation

Documentation is as important as code! Ways to contribute:

- Fix typos or clarify existing docs
- Add missing documentation
- Create tutorials or guides
- Translate documentation (future)
- Improve code comments and docstrings

---

## 🔄 Development Workflow

### Branch Strategy

We follow **Git Flow**:

```
main
  ├── develop (default branch for PRs)
  │   ├── feature/your-feature-name
  │   ├── bugfix/issue-123
  │   └── docs/improve-readme
  └── hotfix/critical-bug (only for prod fixes)
```

**Branch naming conventions:**

- `feature/short-description` - New features
- `bugfix/issue-123` - Bug fixes (reference issue number)
- `docs/what-you-updated` - Documentation
- `test/what-you-tested` - Test improvements
- `refactor/what-you-refactored` - Code refactoring
- `perf/what-you-optimized` - Performance improvements

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Other changes (dependencies, etc.)

**Examples:**

```bash
feat(whisper): add support for large-v3-turbo model

Implemented Whisper large-v3-turbo for faster transcription.
Reduces processing time by 4x compared to large-v3.

Closes #123
```

```bash
fix(api): handle missing audio file gracefully

Previously, missing audio files caused 500 errors.
Now returns 404 with helpful error message.

Fixes #456
```

```bash
docs(readme): add deployment section

Added comprehensive deployment guide including:
- Docker Compose
- Kubernetes
- Cloud providers (AWS, GCP, Azure)
```

### Development Cycle

1. **Create a branch**

   ```bash
   git checkout develop
   git pull upstream develop
   git checkout -b feature/your-feature
   ```

2. **Make changes**

   - Write code
   - Add/update tests
   - Update documentation
   - Run linters and tests locally

3. **Commit changes**

   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

4. **Keep your branch updated**

   ```bash
   git fetch upstream
   git rebase upstream/develop
   ```

5. **Push to your fork**

   ```bash
   git push origin feature/your-feature
   ```

6. **Create Pull Request**

   - Go to GitHub
   - Click "New Pull Request"
   - Select `develop` as base branch
   - Fill out PR template
   - Request review

---

## 💻 Coding Standards

### Python Style Guide

We follow **PEP 8** with some extensions:

- **Line length**: 100 characters (not 79)
- **Docstrings**: Google style
- **Type hints**: Required for public functions
- **Imports**: Organized with `isort`

### Formatting Tools

**Black** (code formatter):
```bash
black app/ tests/
```

**Ruff** (linter):
```bash
ruff check app/ tests/
```

**isort** (import sorting):
```bash
isort app/ tests/
```

**mypy** (type checking):
```bash
mypy app/
```

**Run all checks:**
```bash
make lint  # or
pre-commit run --all-files
```

### Code Structure

**File organization:**

```python
"""Module docstring explaining purpose.

Example:
    from app.models import AudioModel
    
    audio = AudioModel(path="audio.mp3")
"""

# Standard library imports
import os
from typing import Optional, List

# Third-party imports
import numpy as np
from pydantic import BaseModel

# Local imports
from app.utils import helper_function

# Constants
MAX_AUDIO_DURATION = 14400  # 4 hours in seconds

# Classes and functions
class AudioProcessor:
    """Process audio files for transcription.
    
    Attributes:
        model_name: Name of the Whisper model to use.
        device: Device to run inference on ('cpu', 'cuda').
    
    Example:
        >>> processor = AudioProcessor(model_name="large-v3-turbo")
        >>> result = processor.transcribe("audio.mp3")
    """
    
    def __init__(self, model_name: str = "large-v3-turbo", device: str = "cpu"):
        """Initialize the audio processor.
        
        Args:
            model_name: Whisper model variant.
            device: Computation device.
        """
        self.model_name = model_name
        self.device = device
    
    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file.
        
        Returns:
            Dictionary with 'text', 'segments', 'language'.
        
        Raises:
            FileNotFoundError: If audio file doesn't exist.
            ValueError: If audio format is unsupported.
        """
        # Implementation
        pass
```

### Docstring Format (Google Style)

```python
def function_name(param1: str, param2: int = 10) -> dict:
    """One-line summary.
    
    Longer description if needed. Explain what the function does,
    any important details, algorithms used, etc.
    
    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 10.
    
    Returns:
        Description of return value.
        
    Raises:
        ValueError: If param1 is empty.
        TypeError: If param2 is not an integer.
    
    Example:
        >>> result = function_name("test", 20)
        >>> print(result)
        {'key': 'value'}
    """
    pass
```

### Type Hints

**Required for:**
- All public functions
- Function parameters
- Return types

**Example:**

```python
from typing import List, Optional, Dict, Any

def process_audios(
    audio_paths: List[str],
    model: str = "whisper-large-v3-turbo",
    options: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Process multiple audio files."""
    if options is None:
        options = {}
    # Implementation
    return []
```

---

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Tests with dependencies (DB, Redis)
├── e2e/            # End-to-end tests (full pipeline)
└── fixtures/       # Test data and fixtures
```

### Writing Tests

**Unit test example:**

```python
import pytest
from app.processors.whisper import WhisperProcessor

class TestWhisperProcessor:
    """Tests for WhisperProcessor class."""
    
    def test_initialization(self):
        """Test processor initializes correctly."""
        processor = WhisperProcessor(model_name="tiny")
        assert processor.model_name == "tiny"
        assert processor.device == "cpu"
    
    def test_transcribe_valid_audio(self, sample_audio_file):
        """Test transcription of valid audio file."""
        processor = WhisperProcessor()
        result = processor.transcribe(sample_audio_file)
        
        assert "text" in result
        assert "segments" in result
        assert len(result["text"]) > 0
    
    def test_transcribe_missing_file_raises_error(self):
        """Test error handling for missing file."""
        processor = WhisperProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.transcribe("nonexistent.mp3")
```

**Using fixtures:**

```python
# conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    audio_path = tmp_path / "test_audio.mp3"
    # Create dummy audio file
    audio_path.write_bytes(b"fake audio content")
    return str(audio_path)

@pytest.fixture
def db_session():
    """Provide a test database session."""
    # Setup test database
    session = create_test_db_session()
    yield session
    # Cleanup
    session.close()
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_whisper.py

# Specific test
pytest tests/unit/test_whisper.py::TestWhisperProcessor::test_transcribe_valid_audio

# With coverage
pytest --cov=app --cov-report=html

# Parallel execution (faster)
pytest -n auto

# Verbose
pytest -v -s

# Stop on first failure
pytest -x
```

### Coverage Requirements

- **Unit tests**: >80% coverage
- **Integration tests**: Critical paths covered
- **E2E tests**: Main user workflows covered

**Check coverage:**

```bash
pytest --cov=app --cov-report=term-missing
# View HTML report
open htmlcov/index.html
```

---

## 📚 Documentation

### Code Documentation

**All public APIs must have:**
- Module docstring
- Class docstring
- Function/method docstrings with Args, Returns, Raises
- Example usage

**Inline comments:**
- Explain **why**, not **what**
- Use for complex logic
- Keep concise

```python
# Good: Explains why
# Use exponential backoff to avoid overwhelming the API
retry_delay = 2 ** attempt

# Bad: States the obvious
# Set retry_delay to 2 to the power of attempt
retry_delay = 2 ** attempt
```

### Documentation Files

When adding/changing features, update:

- `README.md` - If it affects quick start or main features
- `docs/guides/USER_GUIDE.md` - If it's a user-facing feature
- `docs/guides/API_REFERENCE.md` - If you change API
- `CHANGELOG.md` - Always add entry for your change

### Writing Style

- **Clear and concise**: Short sentences, active voice
- **Examples**: Show, don't just tell
- **User perspective**: Write for the reader (user/developer)
- **Consistent terminology**: Use same terms throughout

---

## 🔀 Pull Request Process

### Before Submitting PR

**Checklist:**

- [ ] Code follows project style guide
- [ ] All tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] Commits follow conventional commit format
- [ ] Branch is up to date with `develop`
- [ ] No merge conflicts

### PR Template

When you create a PR, fill out this template:

```markdown
## Description

Brief description of what this PR does.

Fixes #(issue_number)

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?

Describe the tests you ran to verify your changes.

- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Screenshots (if applicable)

Add screenshots to demonstrate UI changes.
```

### Review Process

1. **Automated checks** run (CI/CD)
   - Linting (Black, Ruff, mypy)
   - Tests (unit, integration)
   - Coverage check
   - Security scan

2. **Code review** by maintainers
   - At least 1 approval required
   - Address review comments
   - Make requested changes

3. **Merge**
   - Squash and merge (for clean history)
   - Delete branch after merge

### Response Time Expectations

- **Initial review**: Within 2 business days
- **Follow-up reviews**: Within 1 business day
- **Merging**: Same day as final approval

---

## 🌍 Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas, show & tell
- **Slack** (if available): Real-time chat
- **Email**: [your.email@example.com]

### Getting Help

**Stuck on something?**

1. Check existing documentation
2. Search [GitHub Issues](https://github.com/yourname/audiomind/issues)
3. Ask in [GitHub Discussions](https://github.com/yourname/audiomind/discussions)
4. Reach out on Slack

**Don't be shy!** No question is too basic. We're here to help.

### Recognition

We recognize contributors in:

- `CONTRIBUTORS.md` file
- Release notes
- Project README
- Special shout-outs for major contributions

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers the project.

---

## 🙏 Thank You!

Your contributions, no matter how small, make a big difference. We appreciate:

- Bug reports and fixes
- Feature implementations
- Documentation improvements
- Answering questions
- Reviewing pull requests
- Sharing the project

**Thank you for being part of AudioMind!** 🎉

---

## 📚 Additional Resources

- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Python PEP 8](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)

---

**Questions?** Open an issue or discussion, we're happy to help!
