import json
import datetime

class RAGLogger:

    def __init__(self, logfile="rag_logs.jsonl"):
        self.logfile = logfile

    def log(self, data):

        entry = {
            "timestamp": str(datetime.datetime.now()),
            "data": data
        }

        with open(self.logfile, "a") as f:
            f.write(json.dumps(entry) + "\n")