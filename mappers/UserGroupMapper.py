#!/usr/bin/python
# -*- coding: utf-8 -*-

class UserGroupMapper(object):
    def __init__(self, db_session):
        self.db_session = db_session

    def get_user_group_using_id(self, user_group_id):
        result_proxy = self.db_session.execute('SELECT * FROM spf_user_groups WHERE user_group_id = :user_group_id', {
            'user_group_id': user_group_id
        }).fetchone()
        return result_proxy

    def get_user_group_using_slug(self, user_group_slug):
        result_proxy = self.db_session.execute('SELECT * FROM spf_user_groups WHERE user_group_slug = :user_group_slug', {
            'user_group_slug': user_group_slug
        }).fetchone()
        return result_proxy
