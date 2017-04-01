#!/usr/bin/python
# -*- coding: utf-8 -*-
from models.User import User

class UserMapper(object):
    def __init__(self, db_session):
        self.db_session = db_session

    def get_user_using_user_id(self, user_id):
        return self.db_session.query(User).filter(User.user_id == user_id).first()

    def get_user_using_username(self, username):
        return self.db_session.query(User).filter(User.username == username).first()

    def get_user_using_email(self, email):
        return self.db_session.query(User).filter(User.email == email).first()
