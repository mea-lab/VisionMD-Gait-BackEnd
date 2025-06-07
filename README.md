# VisionMD_Backend

## Introduction

**VisionMD** is a software tool for quantifying motor symptoms from video recordings. This repository contains the **BackEnd** component, along with a **static version of the FrontEnd**. You cannot modify the UI from this branch. To modify both the FrontEnd and BackEnd, use the [main branch](https://github.com/mea-lab/VideoAnalysisToolBackend/tree/main).

---

## Local Setup Instructions

To run the project locally, you’ll use **Anaconda** to manage the Python environment. This application has been tested on **Linux** (Chrome browser only).

### Step-by-Step Setup (All OS)

1. **Install Anaconda**  
   Download and install Anaconda for your operating system from:  
   [https://www.anaconda.com/download](https://www.anaconda.com/download)

2. **Clone the repository**

   ```bash
   git clone https://github.com/<your-repo-path>/VisionMD_Backend.git
   cd VisionMD_Backend
   ```

3. **Create and activate a virtual environment with Python 3.10**

   ```bash
   conda create --name VisionMD python=3.10
   conda activate VisionMD
   ```

4. **Install the required Python packages**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the development server**

   ```bash
   python manage.py runserver
   ```

6. **Access the application**

   Open **Google Chrome** and go to:  
   [http://127.0.0.1:8000](http://127.0.0.1:8000)

7. **To stop the server**, press `Ctrl + C` in the terminal.

---

## Docker Setup (Optional)

If you prefer using Docker:

1. **Build the Docker containers**

   ```bash
   docker compose build
   ```

2. **Start the services**

   ```bash
   docker compose up -d
   ```

3. **Stop and clean up**

   ```bash
   docker compose down --volumes
   ```

---

> ⚠️ **Note:** The application is tested only with **Google Chrome**. Other browsers may not be fully supported.
