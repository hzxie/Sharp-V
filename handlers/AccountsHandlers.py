#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging

from json import dumps as dump_json
from tornado.web import asynchronous

from handlers.BaseHandler import BaseHandler
from mappers.UserMapper import UserMapper

class LoginHandler(BaseHandler):
    def initialize(self, db_session):
        self.user_mapper = UserMapper(db_session)

    def get(self):
        is_logged_out   = self.get_argument("logout", default=False, strip=False)
        username        = self.get_secure_cookie('user')

        if username and is_logged_out:
            self.clear_cookie('user')
        
        if not username or is_logged_out:
            self.render('accounts/login.html', is_logged_out=is_logged_out)
        else:
            self.redirect('/')

    @asynchronous
    def post(self):
        username        = self.get_argument("username", default=None, strip=False)
        password        = self.get_argument("password", default=None, strip=False)
        keep_signed_in  = self.get_argument("keepSignedIn", default=False, strip=False)

        is_account_valid    = False
        is_allow_to_access  = True
        if username and password:
            user    = self.get_logged_in_user(username, password)

            if user and user.user_group_slug == 'forbidden':
                is_allow_to_access  = False
            if user and user.user_group_slug != 'forbidden':
                is_account_valid    = True
                self.set_secure_cookie('user', username)

        self.write(dump_json({
            'is_successful': is_account_valid and is_allow_to_access,
            'is_username_empty': False if username else True,
            'is_password_empty': False if password else True,
            'is_account_valid': is_account_valid,
            'is_allow_to_access': is_allow_to_access,
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
    def initialize(self, db_session):
        self.user_mapper = UserMapper(db_session)

    def get(self):
        username        = self.get_secure_cookie('user')

        if not username:
            self.render('accounts/register.html')
        else:
            self.redirect('/')

    @asynchronous
    def post(self):
        username        = self.get_argument("username", default=None, strip=False)
        password        = self.get_argument("password", default=None, strip=False)
        
        self.finish()
