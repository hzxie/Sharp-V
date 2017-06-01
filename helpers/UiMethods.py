#!/usr/bin/python
# -*- coding: utf-8 -*-

def base_url(self, url):
    return '%s%s' % (self.application.settings['base_url'], url)

def static_url(self, url):
    return '%s/static/%s' % (self.application.settings['base_url'], url)
