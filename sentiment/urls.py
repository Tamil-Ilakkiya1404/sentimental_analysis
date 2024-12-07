# sentiment/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_sentiment, name='predict_sentiment'),
]
