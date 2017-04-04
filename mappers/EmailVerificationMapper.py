#!/usr/bin/python
# -*- coding: utf-8 -*-

class EmailVerificationMapper(object):
	def __init__(self, db_session):
		self.db_session=db_session

	def get_email_and_token_using_email(self, email):
		result_proxy = self.db_session.execute('SELECT * FROM sharpv_email_verification WHERE email = :email', {
			'email': email
		}).fetchone()
		return result_proxy

	def create_email_verification(self, email, token, expire_time):
		rows_affected = self.db_session.execute('INSERT INTO sharpv_email_verification VALUES (:email, :token, :expire_time)', {
			'email': email,
			'token': token,
			'expire_time': expire_time
		}).rowcount
		return rows_affected

	def delete_email_verification(self, email):
		rows_affected = self.db_session.execute('DELETE FROM sharpv_email_verification WHERE email = :email', {
			'email': email
		}).rowcount
		return rows_affected
