# Modly Installation Guide (Windows)

This guide provides a clean, step-by-step installation process for Windows users using PowerShell.

---

## 🚀 1. Core Installation

Open a **standard PowerShell** window (you do not need Administrator for this part).

### Step A: Clone the Repo
```powershell
git clone https://github.com/lightningpixel/modly.git
cd "$HOME\modly\api"
```

### Step B: Set up Python Environment
```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 📦 2. Frontend Setup & Launch

### Step A: Install Node.js Dependencies
Ensure you have **Node.js** installed.
```powershell
cd "$HOME\modly"
npm install
```
For Textures, we need the CUDA toolkit since it doesnt just use python directly: 
CUDA Toolkit 12.4 from NVIDIA:
https://developer.nvidia.com/cuda-12-4-0-download-archive

### Step B: Launch
The easiest way to run Modly is using the included batch file:
```powershell
cmd /c launch.bat
```

---

## 🛠️ Troubleshooting & Requirements

### 🐍 If Python is missing:
If you see *"Python was not found"*, run this command, then **restart PowerShell**:
```powershell
winget install Python.Python.3.11 --override "/quiet InstallAllUsers=1 PrependPath=1"
```

### 🟢 If Node.js (npm) is missing:
If `npm install` fails, run this command, then **restart PowerShell**:
```powershell
winget install OpenJS.NodeJS
```

### 🏗️ If "Something went wrong" (Bundled Python):
If the app can't find its internal Python files, run this helper script:
```powershell
cd "$HOME\modly"
node scripts/download-python-embed.js
```

---

## 🔌 Extensions (Hunyuan3D)

### Installation Steps:
1. **Verify Git**: Ensure `C:\Program Files\Git\cmd` is in your System Path.
2. **Permissions**: Do **not** install Modly in a OneDrive folder. This causes permission errors.
3. **Download Weights**: Open the Modly extensions panel and click the **Purple Download Button**. 
   - **Crucial**: Stay on the tab until finished. Restart Modly after the download completes.
4. **Missing Models**: If components are present but "not found," install the [VC Redistributable](https://aka.ms).

---

## 💻 Hardware & Performance
* **VRAM**: 6GB (Minimum) | 8GB+ (Recommended).
* **Efficiency**: The **Turbo** model is more memory-efficient than Standard.
* **Updates**: Currently, multi-image input is being patched. Until then, the system defaults to a single front-view image.

---

### 📝 Developer Notes
See `MODLY_CORE_NOTES.md` for details on preserving named multi-image inputs. The extension is pre-configured to handle `left`, `back`, and `right` image paths once the core mapping is updated.
