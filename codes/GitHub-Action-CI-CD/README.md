# GitHub-Action-CI-CD

A simple, containerized Flask application demonstrating a robust CI/CD pipeline using GitHub Actions. This repository serves as a template for automating the building, testing, and deployment of Python-based microservices.

---

## 🚀 Features

- **Flask Web App**: A minimal Python web server.
- **Automated Testing**: Unit tests implemented with `pytest`.
- **Dockerized**: Containerized for consistent environments across development and production.
- **CI/CD Pipeline**: 
    - **Continuous Integration (CI)**: Automatically runs tests on every push or pull request.
    - **Continuous Deployment (CD)**: Automatically builds and pushes a Docker image to Docker Hub upon successful CI.

---

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **Framework**: Flask
- **Testing**: Pytest
- **Containerization**: Docker
- **Automation**: GitHub Actions

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- [Python 3.9+](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/get-docker/)

---

## 🤖 CI/CD Workflow

This project uses GitHub Actions (defined in [cicd.yml](.github/workflows/cicd.yml)) to automate the development lifecycle.

### Workflow Steps:
1. **CI Job (`build-and-test`)**:
    - Checks out the code.
    - Sets up Python 3.10.
    - Installs dependencies.
    - Runs `pytest`.
2. **CD Job (`build-and-publish`)**:
    - Triggered only if the CI job succeeds.
    - Logs into Docker Hub.
    - Builds and pushes the Docker image to Docker Hub.

### Required Secrets
To make the CD job work, you need to add the following secrets to your GitHub repository (`Settings > Secrets and variables > Actions`):
- `DOCKER_USERNAME`: Your Docker Hub username.
- `DOCKER_PASSWORD`: Your Docker Hub password or Personal Access Token.

---

## 📂 Project Structure
```text
.
├── .github/workflows/
│   └── cicd.yml         # GitHub Actions workflow
├── app.py               # Main Flask application
├── test_app.py          # Unit tests
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
└── .gitignore           # Files to ignore in Git
```

---

## 🏃 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/SiddhuShkya/GitHub-Action-CI-CD.git
cd GitHub-Action-CI-CD
```

### 2. Local Setup (Standard)
It's recommended to use a virtual environment:
```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the App
```bash
python app.py
```
The application will be available at `http://localhost:5000`.

### 4. Run Tests
```bash
pytest
```

---

## 🐳 Docker Usage

### Build the Image
```bash
docker build -t flask-test-app .
```

### Run the Container
```bash
docker run -p 5000:5000 flask-test-app
```

---


