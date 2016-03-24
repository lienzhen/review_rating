from __future__ import unicode_literals

from django.db import models
from users.models import User

class Shop(models.Model):
    shop_id = models.CharField(max_length=20, unique=True, primary_key=True)
    title = models.CharField(max_length=300)
    star = models.FloatField()
    star_1_num = models.IntegerField()
    star_2_num = models.IntegerField()
    star_3_num = models.IntegerField()
    star_4_num = models.IntegerField()
    star_5_num = models.IntegerField()
    review_num = models.IntegerField()
    average = models.FloatField()
    taste_score = models.FloatField()
    envir_score = models.FloatField()
    service_score = models.FloatField()
    telephone = models.CharField(max_length=50)
    address = models.CharField(max_length=500)

    class Meta:
        db_table = "shop"

class Recommendation(models.Model):
    shop = models.ForeignKey(Shop)
    user = models.ForeignKey(User)
    score = models.FloatField()
