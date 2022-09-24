#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 9/22/22
# description : A developmental script to run the static documentation site

from livereload import Server
from livereload import shell

if __name__ == '__main__':
    server = Server()
    server.watch('*.rst', shell('make -C ../ html'), delay=1)
    server.watch('*.md', shell('make -C ../ html'), delay=1)
    server.watch('*.py', shell('make -C ../ html'), delay=1)
    server.watch('../../text_dedup/**/*.py', shell('make -C ../ html'), delay=1)
    server.watch('_static/*', shell('make -C ../ html'), delay=1)
    server.watch('_templates/*', shell('make -C ../ html'), delay=1)
    server.serve(root='../build/html')
