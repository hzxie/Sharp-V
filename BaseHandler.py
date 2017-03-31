#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging

from datetime import datetime
from tornado.web import RequestHandler
from tornado.locale import get as get_locale

class BaseHandler(RequestHandler):
    """ The base handlers for all handlers.
    """
    def _handle_request_exception(self, exception):
        """ Error handler of the application.

        Args:
            self: the BaseHandler itself
            exception: the exception pass to this handler
        """
        exception_type_name = type(exception).__name__
        if exception_type_name == 'HTTPError':
            self.set_status(exception.status_code)
            logging.warn('HTTPError[Code=%d] occurred: %s.' % \
                (exception.status_code, exception.log_message))
        else:
            self.set_status(500)
            logging.error('Exception occurred: %s' % exception)
        
        self.render('default/error.html', status_code=self.get_status())

    def get_user_locale(self):
        """Get language settings of users.

        Args:
            self: the BaseHandler itself
        """
        user_locale = self.get_cookie('user_locale')

        if not user_locale:
            # Fallback to browser based locale detection
            user_locale = self.get_browser_locale().code
            self.set_cookie('user_locale', user_locale)

        return get_locale(user_locale)

    def get_formated_date(self, date):
        """Get fotmatted date string for different locales.

        Args:
            self: the BaseHandler itself
            date: the datetime object to be formatted

        Returns:
            the fotmatted date string
        """
        DATE_FORMATS = {
            'en_US': '%b %d, %Y',
            'zh_CN': '%Y年%m月%d日',
        }
        user_locale = self.get_user_locale().code
        dateObj = datetime.strptime(date, "%Y-%m-%d")
        return datetime.strftime(dateObj, DATE_FORMATS[user_locale])
