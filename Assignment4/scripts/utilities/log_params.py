import logging

# set log level
log = logging.getLogger()
log.setLevel(logging.INFO)  # set to logging.DEBUG for more detailed output
for handler in log.handlers[:]: # Remove existing handlers
    log.removeHandler(handler)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt='%(asctime)s - %(message)s', 
    datefmt="%H:%M:%S" 
    )
handler.setFormatter(formatter)
log.addHandler(handler)