#!/usr/bin/python
# -*- coding: utf-8 -*-
from tornado.web import asynchronous

from handlers.BaseHandler import BaseHandler

class HomeHandler(BaseHandler):
    """The default handler for the application."""
    @asynchronous
    def get(self):
        self.render('default/homepage.html')

class AboutHandler(BaseHandler):
    """The handler redirect to about page."""
    @asynchronous
    def get(self):
        self.render('default/about.html')

class TutorialHandler(BaseHandler):
    """The handler redirect to about page."""
    @asynchronous
    def get(self):
        self.render('default/tutorial.html')

class SetLocaleHandler(BaseHandler):
    """The handler of setting locates"""
    @asynchronous
    def get(self):
        user_locale = self.get_argument('user_locale')
        self.set_cookie('user_locale', user_locale)
        self.finish({
            'is_successful': True
        })

class UpgradeBrowserHandler(BaseHandler):
    """ he handler redirect to upgrade browser page."""
    @asynchronous
    def get(self):
        self.render('default/upgrade-browser.html')
