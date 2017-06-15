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
from tornado import template
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
from tornado.locale import load_gettext_translations
from tornado.options import define
from tornado.options import options
from tornado.options import parse_config_file
from tornadomail.backends.smtp import EmailBackend

from handlers.BaseHandler import BaseHandler
from handlers.AccountsHandlers import ForgotPasswordHandler
from handlers.AccountsHandlers import LoginHandler
from handlers.AccountsHandlers import ProfileHandler
from handlers.AccountsHandlers import ProjectsHandler
from handlers.AccountsHandlers import RegisterHandler
from handlers.AccountsHandlers import ResetPasswordHandler
from handlers.DefaultHandlers import AboutHandler
from handlers.DefaultHandlers import TutorialHandler
from handlers.DefaultHandlers import HomeHandler
from handlers.DefaultHandlers import SetLocaleHandler
from handlers.DefaultHandlers import UpgradeBrowserHandler
from handlers.WorkbenchHandlers import DatasetProcessHandler
from handlers.WorkbenchHandlers import DatasetSuggestionsHandler
from handlers.WorkbenchHandlers import DatasetUploadHandler
from handlers.WorkbenchHandlers import WorkbenchHandler
from handlers.WorkbenchHandlers import DatasetUploadHandler
from helpers import UiMethods

class Application(tornado.web.Application):
    def __init__(self, base_url, db_session, mail_sender):
        """ The constructor of Tornado Application.

        Args:
            self: The Application itself.
            db_session: The connection of MySQL
            mail_sender: The mail sender for the application
        """
        handlers = [
            (r"/", HomeHandler),
            (r"/accounts/forgot-password", ForgotPasswordHandler, 
                dict(db_session=db_session, mail_sender=mail_sender)),
            (r"/accounts/login", LoginHandler, dict(db_session=db_session)),
            (r"/accounts/register", RegisterHandler, dict(db_session=db_session)),
            (r"/accounts/reset-password", ResetPasswordHandler, dict(db_session=db_session)),
            (r"/accounts/profile", ProfileHandler, dict(db_session=db_session)),
            (r"/accounts/projects", ProjectsHandler, dict(db_session=db_session)),
            (r"/datasets/suggestions", DatasetSuggestionsHandler),
            (r"/datasets/upload", DatasetUploadHandler),
            (r"/datasets/process", DatasetProcessHandler),
            (r"/workbench", WorkbenchHandler),
            (r"/tutorial", TutorialHandler),
            (r"/about", AboutHandler),
            (r"/set-locale", SetLocaleHandler),
            (r"/not-supported", UpgradeBrowserHandler),
        ]
        settings = {
            'base_url': base_url,
            'cookie_secret': '__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__',
            'debug': True,
            'default_handler_class': BaseHandler,
            'login_url': '/accounts/login',
            'template_path': join(dirname(__file__), 'templates'),
            'static_path': join(dirname(__file__), "static"),
            'xsrf_cookies': True,
            'compress_response': True,
            'ui_methods': UiMethods
        }
        tornado.web.Application.__init__(self, handlers, **settings)

def main():
    """ The entrance of the application."""
    define('base_url', default='https://mlg.hit.edu.cn/spf', help='The URL of the application', type=str)
    define('http_port', default=8000, help='The port of the application', type=int)
    define('mysql_host', default='127.0.0.1:3306', help='The host of MySQL', type=str)
    define('mysql_database', default='sharpv', help='The database name of MySQL', type=str)
    define('mysql_username', default='root', help='The user of MySQL', type=str)
    define('mysql_password', default='', help='The password of MySQL', type=str)
    define('mysql_pool_size', default=10, help='The pool size of MySQL', type=int)
    define('mysql_pool_recycle', default=5, help='The pool recycle of MySQL', type=int)
    define('mail_host', default='', help='The host of mail server', type=str)
    define('mail_port', default=587, help='The port of mail server', type=int)
    define('mail_username', default='', help='The username of mail server', type=str)
    define('mail_password', default='', help='The password of mail server', type=str)
    define('dataset_directory', default='/var/datasets', help="The path of original dataset files", type=str)
    parse_config_file(join(dirname(__file__), 'server.conf'))

    # Setup MySQL connection
    db_engine   = create_engine('mysql://%s:%s@%s/%s?charset=utf8' %
                        (options.mysql_username, options.mysql_password, 
                            options.mysql_host, options.mysql_database),
                        encoding='utf-8', echo=False, pool_size=options.mysql_pool_size, 
                        pool_recycle=options.mysql_pool_recycle)
    db_session  = scoped_session(sessionmaker(bind=db_engine,
                        autocommit=True, autoflush=True, expire_on_commit=False))

    # Setup mail sender
    mail_sender = EmailBackend(
        options.mail_host, options.mail_port, options.mail_username, options.mail_password, True,
        template_loader=template.Loader(join(dirname(__file__), 'templates', 'mails'))
    )

    # Load translations for different locale
    # load_gettext_translations(join(dirname(__file__), 'locales'), 'messages')

    # Start HTTP Server
    http_server = HTTPServer(Application(options.base_url, db_session, mail_sender), max_buffer_size=10485760000)
    http_server.listen(options.http_port)
    logging.info('Sharp-V web application is running on port: %d.' % options.http_port)
    IOLoop.current().start()

if __name__ == "__main__":
    main()