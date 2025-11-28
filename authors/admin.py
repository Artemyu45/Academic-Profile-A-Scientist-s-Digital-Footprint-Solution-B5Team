from django.contrib import admin
from .models import Author

@admin.register(Author)
class AuthorAdmin(admin.ModelAdmin):
    list_display = ('full_name', 'affiliation', 'citation_count', 'h_index')
    search_fields = ('full_name', 'affiliation')
