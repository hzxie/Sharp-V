#!/usr/bin/python
# -*- coding: utf-8 -*-

class UserMapper(object):
    def __init__(self, db_session):
        self.db_session = db_session

    def get_user_using_user_id(self, user_id):
        result_proxy =  self.db_session.execute('SELECT * FROM sharpv_users NATURAL JOIN sharpv_user_groups WHERE user_id = :user_id', {
            'user_id': user_id
        }).fetchone()
        return result_proxy

    def get_user_using_username(self, username):
        result_proxy = self.db_session.execute('SELECT * FROM sharpv_users NATURAL JOIN sharpv_user_groups WHERE username = :username', {
            'username': username
        }).fetchone()
        return result_proxy

    def get_user_using_email(self, email):
        result_proxy = self.db_session.execute('SELECT * FROM sharpv_users NATURAL JOIN sharpv_user_groups WHERE email = :email', {
            'email': email
        }).fetchone()
        return result_proxy

    def create_user(self, username, password, email, user_group_id):
        row_count = self.db_session.execute('INSERT INTO sharpv_users (username, password, email, user_group_id) VALUES (:username, md5(:password), :email, :user_group_id)', {
            'username': username,
            'password': password,
            'email': email,
            'user_group_id': user_group_id
        }).rowcount
        return row_count

class UserGroupMapper(object):
    def __init__(self, db_session):
        self.db_session = db_session

    def get_user_group_using_id(self, user_group_id):
        result_proxy = self.db_session.execute('SELECT * FROM sharpv_user_groups WHERE user_group_id = :user_group_id', {
            'user_group_id': user_group_id
        }).fetchone()
        return result_proxy

    def get_user_group_using_slug(self, user_group_slug):
        result_proxy = self.db_session.execute('SELECT * FROM sharpv_user_groups WHERE user_group_slug = :user_group_slug', {
            'user_group_slug': user_group_slug
        }).fetchone()
        return result_proxy
