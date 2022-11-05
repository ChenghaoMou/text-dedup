#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date         : 2021-06-05 12:48:33
# @Author       : Chenghao Mou (mouchenghao@gmail.com)

"""Text deduplication simplified."""

import logging

from rich.logging import RichHandler

logger = logging.getLogger("text_dedup")
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler(rich_tracebacks=True))
logger.propagate = False
