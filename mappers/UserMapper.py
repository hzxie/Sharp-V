#!/usr/bin/python
# -*- coding: utf-8 -*-

class UserMapper(object):
    def __init__(self, db_session):
        self.db_session = db_session

    def get_user_using_user_id(self, user_id):
        result_proxy =  self.db_session.execute('SELECT * FROM spf_users NATURAL JOIN spf_user_groups WHERE user_id = :user_id', {
            'user_id': user_id
        }).fetchone()
        return result_proxy

    def get_user_using_username(self, username):
        result_proxy = self.db_session.execute('SELECT * FROM spf_users NATURAL JOIN spf_user_groups WHERE username = :username', {
            'username': username
        }).fetchone()
        return result_proxy

    def get_user_using_email(self, email):
        result_proxy = self.db_session.execute('SELECT * FROM spf_users NATURAL JOIN spf_user_groups WHERE email = :email', {
            'email': email
        }).fetchone()
        return result_proxy

    def create_user(self, username, password, email, user_group_id):
        rows_affected = self.db_session.execute('INSERT INTO spf_users (username, password, email, user_group_id) VALUES (:username, :password, :email, :user_group_id)', {
            'username': username,
            'password': password,
            'email': email,
            'user_group_id': user_group_id
        }).rowcount
        return rows_affected

    def update_password_using_email(self, email, password):
        rows_affected = self.db_session.execute('UPDATE spf_users SET password = :password WHERE email = :email', {
            'email': email,
            'password': password
        }).rowcount
        return rows_affected

    def update_user(self, user):
        rows_affected = self.db_session.execute('UPDATE spf_users SET email = :email, password = :password WHERE user_id = :user_id', user).rowcount
        return rows_affected
