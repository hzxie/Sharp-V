#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import tornado.web

from os.path import dirname
from os.path import join
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
from tornado.locale import load_gettext_translations
from tornado.options import define
from tornado.options import options
from tornado.options import parse_config_file

from BaseHandler import BaseHandler
from DefaultHandlers import AboutHandler
from DefaultHandlers import TermsHandler
from DefaultHandlers import PrivacyHandler
from DefaultHandlers import HomeHandler
from DefaultHandlers import SetLocaleHandler
from DefaultHandlers import UpgradeBrowserHandler

class Application(tornado.web.Application):
    def __init__(self):
        """ The constructor of Tornado Application.

        Args:
            self: The Application itself.
        """
        handlers = [
            (r"/", HomeHandler),
            (r"/about", AboutHandler),
            (r"/terms", TermsHandler),
            (r"/privacy", PrivacyHandler),
            (r"/about", AboutHandler),
            (r"/set-locale", SetLocaleHandler),
            (r"/not-supported", UpgradeBrowserHandler),
        ]
        settings = {
            'cookie_secret': '__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__',
            'debug': True,
            'default_handler_class': BaseHandler,
            'login_url': '/accounts/login',
            'template_path': join(dirname(__file__), "templates"),
            'static_path': join(dirname(__file__), "static"),
            'xsrf_cookies': True,
        }
        tornado.web.Application.__init__(self, handlers, **settings)

def main():
    """ The entrance of the application."""
    define("http_port", default=8888, help="The port for this application", type=int)
    define("dataset_directory", default='/var/datasets', help="The path of original dataset files", type=str)
    parse_config_file(join(dirname(__file__), 'server.conf'))

    # Load translations for different locale
    load_gettext_translations(join(dirname(__file__), 'locales'), 'messages')

    # Start HTTP Server
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.http_port)
    logging.info('BioIndex web application is running on port: %d.' % options.http_port)
    IOLoop.current().start()

if __name__ == "__main__":
    main()