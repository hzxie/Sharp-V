#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import MySQLdb
import tornado.web

from os.path import dirname
from os.path import join
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
from tornado.locale import load_gettext_translations
from tornado.options import define
from tornado.options import options
from tornado.options import parse_config_file

from handlers.BaseHandler import BaseHandler
from handlers.AccountsHandlers import LoginHandler
from handlers.AccountsHandlers import RegisterHandler
from handlers.DefaultHandlers import AboutHandler
from handlers.DefaultHandlers import TermsHandler
from handlers.DefaultHandlers import PrivacyHandler
from handlers.DefaultHandlers import HomeHandler
from handlers.DefaultHandlers import SetLocaleHandler
from handlers.DefaultHandlers import UpgradeBrowserHandler

class Application(tornado.web.Application):
    def __init__(self, db_session):
        """ The constructor of Tornado Application.

        Args:
            self: The Application itself.
            mysqlConnection: The connection of MySQL
        """
        handlers = [
            (r"/", HomeHandler),
            (r"/accounts/login", LoginHandler),
            (r"/about", AboutHandler),
            (r"/terms", TermsHandler),
            (r"/privacy", PrivacyHandler),
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
            'compress_response': True,
        }
        tornado.web.Application.__init__(self, handlers, **settings)
        self.db_session = db_session

def main():
    """ The entrance of the application."""
    define('http_port', default=8000, help='The port for this application', type=int)
    define('mysql_host', default='127.0.0.1:3306', help='The host of MySQL', type=str)
    define('mysql_database', default='sharpv', help='The database name of MySQL', type=str)
    define('mysql_username', default='root', help='The user of MySQL', type=str)
    define('mysql_password', default='', help='The password of MySQL', type=str)
    define('mysql_pool_size', default=10, help='The pool size of MySQL', type=int)
    define('mysql_pool_recycle', default=5, help='The pool recycle of MySQL', type=int)
    define('dataset_directory', default='/var/datasets', help="The path of original dataset files", type=str)
    parse_config_file(join(dirname(__file__), 'server.conf'))

    # Setup MySQL Connection
    db_engine   = create_engine('mysql://%s:%s@%s/%s?charset=utf8' %
                        (options.mysql_username, options.mysql_password, options.mysql_host, options.mysql_database),
                        encoding='utf-8', echo=False, pool_size=options.mysql_pool_size, 
                        pool_recycle=options.mysql_pool_recycle)
    db_session  = scoped_session(sessionmaker(bind=db_engine,
                        autocommit=True, autoflush=True, expire_on_commit=False))

    # Load translations for different locale
    load_gettext_translations(join(dirname(__file__), 'locales'), 'messages')

    # Start HTTP Server
    http_server = tornado.httpserver.HTTPServer(Application(db_session))
    http_server.listen(options.http_port)
    logging.info('BioIndex web application is running on port: %d.' % options.http_port)
    IOLoop.current().start()

if __name__ == "__main__":
    main()