"""fsr URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from shops import views as shops_views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^q/(?P<user_id>\d+)/$', shops_views.query_recommendations),
    url(r'^random/$', shops_views.render_random),
    #url(r'^image/(?P<filename>.*)/$', django.views.static.serve, {'document_root': '/home/zhangbaihan/lizz/demo/fsr/static/image'}),
]
