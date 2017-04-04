#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging

from hashlib import md5
from json import dumps as dump_json
from re import match as match_regx
from tornado.web import asynchronous

from handlers.BaseHandler import BaseHandler
from mappers.UserMapper import UserMapper
from mappers.UserGroupMapper import UserGroupMapper

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

        result          = self.is_allow_to_access(username, password)
        if result['isSuccessful']:
            self.set_secure_cookie('user', username)
            logging.info('User [Username=%s] logged in at %s.' % (username, self.get_user_ip_addr()))

        self.write(dump_json(result))
        self.finish()

    def is_allow_to_access(self, username, password):
        is_account_valid    = False
        is_allow_to_access  = True
        if username and password:
            user = self.get_user_using_username_or_email(username, password)

            if user:
                is_account_valid = True
                
                if user.user_group_slug == 'forbidden':
                    is_allow_to_access = False

        return {
            'isSuccessful': is_account_valid and is_allow_to_access,
            'isUsernameEmpty': False if username else True,
            'isPasswordEmpty': False if password else True,
            'isAccountValid': is_account_valid,
            'isAllowToAccess': is_allow_to_access,
        }

    def get_user_using_username_or_email(self, username, password):
        user = None

        if username.find('@') == -1:
            user = self.user_mapper.get_user_using_username(username)
        else:
            user = self.user_mapper.get_user_using_email(username)

        if user and user.password == password:
            return user

class RegisterHandler(BaseHandler):
    def initialize(self, db_session):
        self.user_mapper        = UserMapper(db_session)
        self.user_group_mapper  = UserGroupMapper(db_session)

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
        email           = self.get_argument("email", default=None, strip=False)

        result          = self.create_user(username, password, email)
        if result['isSuccessful']:
            logging.info('New user [Username=%s] was created at %s' % (username, self.get_user_ip_addr()))
        
        self.write(dump_json(result))
        self.finish()

    def create_user(self, username, password, email):
        user_group      = self.user_group_mapper.get_user_group_using_slug('users')
        user_group_id   = user_group.user_group_id
        result          = {
            'isUsernameEmpty': False if username else True,
            'isUsernameLegal': True if match_regx(r'^[A-Za-z][A-Za-z0-9_]{5,15}$', username) else False,
            'isUsernameExists': self.is_username_exists(username),
            'isPasswordEmpty': False if password else True,
            'isPasswordLegal': len(password) >= 6 and len(password) <= 16,
            'isEmailEmpty': False if email else True,
            'isEmailLegal': True if match_regx(r'^[A-Za-z0-9\._-]+@[A-Za-z0-9_-]+\.[A-Za-z0-9\._-]+$', email) else False,
            'isEmailExists': self.is_email_exists(email)
        }
        result['isSuccessful'] = not result['isUsernameEmpty']  and     result['isUsernameLegal'] and \
                                 not result['isUsernameExists'] and not result['isPasswordEmpty'] and \
                                     result['isPasswordLegal']  and not result['isEmailEmpty']    and \
                                     result['isEmailLegal']     and not result['isEmailExists']
        if result['isSuccessful']:
            rows_affected = self.user_mapper.create_user(username, md5(password).hexdigest(), email, user_group_id)
            if not rows_affected:
                result['isSuccessful'] = False
        return result

    def is_username_exists(self, username):
        return True if self.user_mapper.get_user_using_username(username) else False

    def is_email_exists(self, email):
        return True if self.user_mapper.get_user_using_email(email) else False
