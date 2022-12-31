from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views

import sys, os
sys.path.insert(1, os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))) + "/DINO_model")
import dino_views

app_name = 'dino'
urlpatterns = [
    # two paths: with or without given image
    path('', views.index, name='index'),
    path('dino_api', dino_views.dino_api, name='dino_api')
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)