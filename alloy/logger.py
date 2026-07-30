import logging
import json
import os
from datetime import datetime, UTC
from pathlib import Path

# Setup logging directory
LOG_DIR = Path("/var/log/panoseti")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(service_name: str) -> logging.Logger:
    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)
    
    # Standard JSONL file handler
    handler = logging.FileHandler(LOG_DIR / f"{service_name}.jsonl")
    
    # Simple JSON formatter
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "service": service_name,
                "level": record.levelname,
                "message": record.getMessage(),
                "hostname": os.getenv("HOSTNAME", "unknown"),
                "pid": record.process,
                "thread": record.threadName,
            }
            # Include extra fields if they exist
            if hasattr(record, "git_commit"):
                log_record["git_commit"] = record.git_commit
            if hasattr(record, "run_id"):
                log_record["run_id"] = record.run_id
                
            return json.dumps(log_record)

    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger

# Example Usage
if __name__ == "__main__":
    logger = get_logger("daq_control")
    logger.info("Service initialized", extra={"git_commit": "abc1234", "run_id": "test_123"})
