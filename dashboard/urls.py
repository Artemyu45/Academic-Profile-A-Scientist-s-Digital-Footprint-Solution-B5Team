from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.home, name='home'),
    path('authors/', views.authors_list, name='authors_list'),
    path('authors/<int:author_id>/', views.author_detail, name='author_detail'),
    path('authors/<int:author_id>/recommendations/', views.recommendations, name='recommendations'),
    path('search/', views.search, name='search'),
]
