# Dashboard Installation Guide

Follow these steps to set up and run the dashboard:

## 1. Ensure Python is Installed

Make sure you have Python installed on your system. You can verify this by running the following command in your terminal:

```bash
python --version
```

## 2. Install Pipenv
Open a terminal in the current directory of the project and install `pipenv` (a tool to manage Python environments and dependencies) by running:

```bash
pip install pipenv
```

## 2. Install Required Python Packages
After installing `pipenv`, execute the following command to create a virtual environment and install all the required Python packages defined in the `Pipfile`:

```bash
pipenv install
```

## 4. Activate the Python Environment
Once the virtual environment is set up, activate it by running:

```bash
pipenv shell
```

This will switch your environment to the virtual one where the required packages are installed.

## 5. Run the Streamlit Dashboard
Finally, to open the Streamlit dashboard, run the following command:

```bash
streamlit run app.py
```

This will launch the dashboard in your default web browser.
