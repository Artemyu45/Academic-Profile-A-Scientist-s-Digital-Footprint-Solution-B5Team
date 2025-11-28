from django.db import models

class Author(models.Model):
    full_name = models.CharField(max_length=128)
    affiliation = models.CharField(max_length=256)
    email = models.EmailField(blank=True, null=True)
    citation_count = models.PositiveIntegerField(default=0)
    h_index = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.full_name
