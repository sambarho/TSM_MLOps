# Streamlit Resume App
This folder contains a simplified version of the streamlit app.py that functions without any llm and comparison functions. It is intended for testing the streamlit interface and database interactions.

## How to run the App:

### 1. Build and Run the Docker Container

Launch your WSL terminal by running:

```bash
wsl
```

> Make sure you're inside your Linux environment (e.g., Ubuntu), not in the Windows file system.

### Step 2: Check Existing Docker Containers

To list all Docker containers (running or stopped):

```bash
sudo docker ps -a
```

### Step 3: Navigate to Your Project Directory

Change to your project folder, e.g.:

```bash
cd ~/Projects/streamlit-resume-app
```

### Step 4: Build the Docker Image

Build the image from your `Dockerfile`:

```bash
sudo docker build -t streamlit-resume-app .
```

### Step 5: Run the Streamlit App

#### Recommended approach for Testing:

* Use this to test test the app
* This will not create any persistent database on your device

```bash
sudo docker run --rm -p 8501:8501 streamlit-resume-app
```

#### Alternative for Development:

* Mounts your local folder inside the container
* Auto-deletes the container when stopped
* Useful for live code changes (But a resume_match.db will be saved to your device, even if its non persistent in the container)

Change The docker file to

```dockerfile
# Use an official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /streamlit-resume-app

# Copy only the requirements first to leverage Docker cache and avoid reinstalling dependencies unnecessarily
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit port (default port for Streamlit)
EXPOSE 8501

# Set the default command (You will override this command with 'docker run' command to run the app in dev mode)
CMD ["streamlit", "run", "simple_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

and then run:

```bash
sudo docker run --rm -p 8501:8501 -v "$PWD":/app -w /app streamlit-resume-app streamlit run simple_app.py
```


### Step 6: Access the App

Open your browser and go to:

```
http://localhost:8501
```

## Added Database Functionality (`database.py`)

This module sets up a simple SQLite database using SQLAlchemy to store resume-job comparison results. Useful for reviewing past comparisons directly in the UI.

### Key Components

* **MatchRecord**: ORM class with fields like resume name, file, job description, parsed info, scores (JSON), and timestamp.
* **save\_match\_record(...)**: Helper function to save comparison results after each match.
* **SQLite setup**: Creates a `resume_match.db` file.

⚠️ **Note:** The database is *not persistent*. Data will be lost when the container stops if run with `--rm`.

### Future Idea:

* Make the data persistent in centralized storage to enable fine-tuning of LLMs based on accumulated comparison data.