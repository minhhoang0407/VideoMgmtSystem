# VideoMgmtSystem
# how to install library

# activate venv
.venv\Scripts\activate
# pip install
pip install -r requirements.txt


# run program
uvicorn main:app --reload

# uvicorn: call library
# main: call main.py
# app: variable API in main.py
# --reload: automatic reload API if u save changes.
search in brower:http://127.0.0.1:8000/docs

