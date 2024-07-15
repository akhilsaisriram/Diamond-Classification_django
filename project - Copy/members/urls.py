from django.urls import path
from .views import *

urlpatterns = [
    path('upload/', ImageUploadView.as_view(), name='image-upload'),
     path('chat/', ChatgptView.as_view()),
    # Other app-specific URLs...
]