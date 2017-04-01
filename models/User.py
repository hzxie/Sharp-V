#!/usr/bin/python
# -*- coding: utf-8 -*-
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    """ Sqlalchemy ORM, mapping ot table an_orm """
    __tablename__ = 'sharpv_users'

    user_id         = Column(Integer, primary_key=True)
    username        = Column(String)
    password        = Column(String)
    email           = Column(String)
    user_group_id   = Column(Integer)