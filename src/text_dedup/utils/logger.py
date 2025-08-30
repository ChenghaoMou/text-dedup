import logging

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(level="ERROR", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

log = logging.getLogger("text_dedup")
log.setLevel(logging.DEBUG)
