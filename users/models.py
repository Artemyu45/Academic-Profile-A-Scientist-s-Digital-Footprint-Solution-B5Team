from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    ROLE_CHOICES = [
        ('researcher', 'Researcher'),
        ('admin', 'Administrator'),
        ('student', 'Student'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    affiliation = models.CharField(max_length=255, blank=True)
    bio = models.TextField(blank=True)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='researcher')

    # API settings
    api_base_url = models.URLField(default='http://localhost:8000/api/')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} - {self.role}"
