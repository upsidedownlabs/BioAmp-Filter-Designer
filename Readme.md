# BioAmp Filter Designer

BioAmp Filter Designer is a Python-based tool that helps you design, visualize, and analyze digital filters (Low-pass, High-pass, Band-pass, Band-stop etc.) used in biomedical signal processing applications such as ECG, EMG, EOG and EEG.

---

## Features

- Interactive GUI for designing filters  
- Supports multiple filter types (Low-pass, High-pass, Band-pass, and Band-stop)  

- Export designed filter for integration into hardware or software projects  

---

## Requirements

Before running the application, ensure you have:

- Python 3.8 or higher installed  
- pip package manager (comes with Python)
- Supported operating systems: Windows / macOS / Linux

---

## Installation & Setup

Follow the steps below to set up and run BioAmp Filter Designer locally:

### 1. Clone the Repository

```bash
git clone https://github.com/upsidedownlabs/BioAmp-Filter-Designer.git
```

### Download as ZIP (alternative)

You can also download the repository as a ZIP file from GitHub:

- Open the repository page on GitHub in your browser.
- Click the "Code" button and choose "Download ZIP".

After downloading, extract the ZIP and open the extracted folder to continue with the steps below.

### 2. Open the Project Directory

Navigate into the repository folder. For example:

```bash
cd BioAmp-Filter-Designer
```

If your downloaded folder has spaces or extra text (for example `BioAmp-Filter-Designer-main (1)`), make sure to navigate into the correct folder path.

From File Explorer (external PowerShell)

Open File Explorer, navigate to the project folder (`d:\bio-amp\BioAmp-Filter-Designer`).
Right-click the folder background and choose "Open in Terminal" or "Open PowerShell window here".

Commands to run (PowerShell)

```powershell
cd 'd:\bio-amp\BioAmp-Filter-Designer'

# Confirm Python is available:
python --version
pip --version


```

### 3. Create a Virtual Environment

Creating a virtual environment ensures dependencies are isolated from your system Python:

```bash
python -m venv .venv
```

This creates a folder named `.venv` containing a clean Python environment.


### 4. Activate the Virtual Environment

On Windows (PowerShell):

```powershell
.venv\Scripts\activate
```

On macOS / Linux:

```bash
source .venv/bin/activate
```

Once activated, your terminal prompt will show `(.venv)` at the beginning — indicating that the environment is active.

### 5. Install Required Dependencies

Install all required Python libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

This installs dependencies needed to run the GUI — such as `tkinter`, `numpy`, `scipy`, and `matplotlib`.

### 6. Run the Application

After installation, run:

```bash
python GUI.py
```

This launches the BioAmp Filter Designer GUI and you can start designing filters interactively.

## Usage Guide

- Select Filter Type — Choose between Low-pass, High-pass, Band-pass, or Band-stop filters.
- Adjust Parameters — Set the cutoff frequency, sampling rate, and filter order.
- Generated files will be saved in the project folder.

---

