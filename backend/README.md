To create a virtual environment and install all necessary packages for your Flask app, you can follow these steps:

1. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**:
   - On **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Install packages**:
   Create a `requirements.txt` file listing all the required packages. You can generate it automatically if you already have packages installed globally:
   ```bash
   pip freeze > requirements.txt
   ```

4. **Install the packages in the virtual environment**:
   ```bash
   pip install -r requirements.txt
   ```

   export KMP_DUPLICATE_LIB_OK=TRUE
python app.py
