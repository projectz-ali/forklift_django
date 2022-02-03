"""URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
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
from django.urls import path
from examples import views
from django.conf import settings
from django.conf.urls.static import static

# from .views_backup import hello_page, callbacks_page, sync_callbacks_page, \
# 				   file_access_page, input_page

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^hello$', views.home),
    # url(r'^hello2$', views.save_data),
    # url(r'^callbacks$', callbacks_page),
    # url(r'^sync_callbacks$', sync_callbacks_page),
    # url(r'^file_access$', file_access_page),
    # url(r'^input$', input_page),
    path('index/',views.index, name='index')


]
# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('',views.home, name='home'),
#     path('index/',views.index, name='index')
# ]
