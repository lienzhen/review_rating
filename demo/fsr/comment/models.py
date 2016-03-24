from __future__ import unicode_literals

from django.db import models
from shops.models import Shop
from users.models import User

class Comment(models.Model):
    comment_id = models.CharField(max_length=20, unique=True, primary_key=True)
    context = models.TextField()
    shop = models.ForeignKey(Shop)
    user = models.ForeignKey(User)
    star = models.IntegerField()

    class Meta:
        db_table = "comment_keyword"

