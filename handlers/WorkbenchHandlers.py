#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging

from handlers.BaseHandler import BaseHandler

class WorkbenchHandler(BaseHandler):
    def get(self):
        self.render('workbench/workbench.html')
