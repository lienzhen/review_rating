#coding=utf-8
from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import render_to_response,redirect
from django.template import RequestContext
from django.template.loader import get_template
from django.views.decorators.csrf import csrf_exempt
from django.core.exceptions import ObjectDoesNotExist
import json
# Create your views here.
from users.models import User
from shops.models import Shop, Recommendation
from comment.models import Comment
import jieba.analyse
import random
import logging
logger = logging.getLogger('django_logger')

colors = ['30A9DE', '9DC8C8', '58C9B9', 'D1B6E1', 'FFEEE4', 'F17F42', 'CE6D39', '30A9DE', 'EFDC05', 'E53A40', 'E0E3DA', '77AF9C', 'F68657', '9055A2', 'FBFFB9', 'EC7357', '548687', '8F2D56', 'E3E36A', 'FFBC42']
def get_recommendations_from_uid(user_id):
    res = []
    recoms = Recommendation.objects.filter(user_id=user_id).order_by('-score')
    for each in recoms:
        res.append((each.shop_id, each.score))
    return res

def render_return(template_file, context):
    template = get_template(template_file)
    html = template.render(context)
    return HttpResponse(html)

def render_random(request):
    recom = Recommendation.objects.all().order_by('?')[:1].get()
    return HttpResponseRedirect('/q/%s' % recom.user_id)

def query_recommendations(request, user_id):
    template_file = 'index.html'
    user = None
    context = {'error': 0, 'err_msg': ''}
    try:
        user = User.objects.get(user_id=user_id)
    except ObjectDoesNotExist:
        context['error'] = 1
        context['err_msg'] = 'User Not Found'
        return render_return(template_file, context)

    recoms = get_recommendations_from_uid(user_id)
    shops = []
    for each in recoms:
        try:
            shop = Shop.objects.get(shop_id=each[0])
        except ObjectDoesNotExist:
            continue
        comment_txt = ''
        comments = Comment.objects.filter(shop_id=each[0])
        for c in comments:
            comment_txt += ' ' + c.context
        keywords = ' '.join([x.encode('utf-8', 'ignore') for x in jieba.analyse.extract_tags(comment_txt, topK=20)])
        #logger.info(keywords)
        s = {}
        s['predict'] = '%.2lf' % each[1]
        s['title'] = shop.title
        s['star'] = int(shop.star)
        s['taste'] = shop.taste_score
        s['envir'] = shop.envir_score
        s['service'] = shop.service_score
        s['telephone'] = shop.telephone
        s['address'] = shop.address
        s['comments'] = comment_txt
        s['keywords'] = keywords
        shops.append(s)
    comment_txt = ''
    comments = Comment.objects.filter(user_id=user_id)
    for each in comments:
        comment_txt += ' ' + each.context
    keywords = [x.encode('utf-8', 'ignore') for x in jieba.analyse.extract_tags(comment_txt, topK=20)]
    kw_rank = []
    for i, kw in enumerate(keywords):
        kw_rank.append({'kw': kw, 'rk': (len(keywords) - i) / 10 * 3, 'cl':'#%s' % random.choice(colors)})
    context['user'] = user
    context['shops'] = shops
    context['comments'] = comment_txt
    context['keywords'] = kw_rank
    return render_return(template_file, context)
