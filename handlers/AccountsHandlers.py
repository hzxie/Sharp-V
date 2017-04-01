#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging

from json import dumps as dump_json
from tornado.web import asynchronous

from handlers.BaseHandler import BaseHandler
from mappers.UserMapper import UserMapper

class LoginHandler(BaseHandler):
    def initialize(self):
        self.user_mapper = UserMapper(self.application.db_session)

    def get(self):
        is_logged_out   = self.get_argument("logout", default=False, strip=False)

        if not self.get_secure_cookie('username'):
            self.render('accounts/login.html', is_logged_out=is_logged_out)
        self.redirect('/')

    @asynchronous
    def post(self):
        username        = self.get_argument("username", default=None, strip=False)
        password        = self.get_argument("password", default=None, strip=False)
        keep_signed_in  = self.get_argument("keepSignedIn", default=False, strip=False)

        user           = None
        if username and password:
            user    = self.get_logged_in_user(username, password)

            # TODO: Check user group
            if user:
                self.set_secure_cookie('username', username)

        self.write(dump_json({
            'is_username_empty': False if username else True,
            'is_password_empty': False if password else True,
            'is_account_valid': True if user else False
        }))
        self.finish()

    def get_logged_in_user(self, username, password):
        user = None

        if username.find('@') == -1:
            user = self.user_mapper.get_user_using_username(username)
        else:
            user = self.user_mapper.get_user_using_email(username)

        if user and user.password == password:
            return user

class RegisterHandler(BaseHandler):
    pass