from __future__ import unicode_literals

from django.db import models

class User(models.Model):
    user_id = models.CharField(max_length=20, unique=True, primary_key=True)
    username = models.CharField(max_length=50)

    class Meta:
        db_table = "user"

