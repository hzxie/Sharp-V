#!/usr/bin/python
# -*- coding: utf-8 -*-
from handlers.BaseHandler import BaseHandler

class HomeHandler(BaseHandler):
    """The default handler for the application."""
    def get(self):
        """Render to homepage.

        Args:
            self: The HomeHandler itself.
        """
        self.render('default/homepage.html')

class AboutHandler(BaseHandler):
    """The handler redirect to about page."""
    def get(self):
        """Render to about page.

        Args:
            self: The HomeHandler itself.
        """
        self.render('default/about.html')

class TermsHandler(BaseHandler):
    """The handler redirect to terms of use page."""
    def get(self):
        """Render to terms of use page.

        Args:
            self: The HomeHandler itself.
        """
        self.render('default/terms.html')

class PrivacyHandler(BaseHandler):
    """ The handler redirect to privacy and cookies page."""
    def get(self):
        """Render to privacy and cookies page.

        Args:
            self: The HomeHandler itself.
        """
        self.render('default/privacy.html')

class SetLocaleHandler(BaseHandler):
    """The handler of setting locates"""
    def get(self):
        user_locale = self.get_argument('user_locale')
        self.set_cookie('user_locale', user_locale)
        self.finish({
            'is_successful': True
        })

class UpgradeBrowserHandler(BaseHandler):
    """ he handler redirect to upgrade browser page."""
    def get(self):
        """Render to upgrade browser page.

        Args:
            self: The HomeHandler itself.
        """
        self.render('default/upgrade-browser.html')