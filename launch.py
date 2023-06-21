#!/usr/bin/env python
import os
from dotenv import load_dotenv
load_dotenv()

SERVER_PORT = os.environ.get("SERVER_PORT", '8000')
SERVER_HOST = os.environ.get("SERVER_HOST", 'localhost')

from ui.interface import webapp

app = webapp()
app.launch(server_name=str(SERVER_HOST),
           server_port=int(SERVER_PORT))