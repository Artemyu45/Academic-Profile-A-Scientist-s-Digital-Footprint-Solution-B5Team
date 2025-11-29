from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from .api_client import FastAPIClient


def get_api_client(request):
    """Get API client with user's base URL if customized"""
    base_url = request.user.profile.api_base_url
    return FastAPIClient(base_url)


@login_required(login_url='users:login')
def home(request):
    try:
        api = get_api_client(request)

        # Get statistics
        top_authors = api.get_top_cited_authors()
        most_cited_papers = api.get_most_cited_publications()

        context = {
            'top_authors': top_authors.get('results', [])[:5],
            'most_cited_papers': most_cited_papers.get('results', [])[:5],
        }
        return render(request, 'dashboard/home.html', context)
    except Exception as e:
        messages.error(request, f"Error connecting to API: {str(e)}")
        return render(request, 'dashboard/home.html', {'error': str(e)})


@login_required(login_url='users:login')
def authors_list(request):
    try:
        api = get_api_client(request)
        search = request.GET.get('search', '')
        page = request.GET.get('page', 1)
        ordering = request.GET.get('ordering', '-citation_count')

        # Get data from API
        data = api.get_authors(page=page, search=search, ordering=ordering)

        # Paginate manually if needed
        authors = data.get('results', [])
        total_count = data.get('count', 0)

        context = {
            'authors': authors,
            'search': search,
            'total_count': total_count,
            'next': data.get('next'),
            'previous': data.get('previous'),
            'page': page,
        }
        return render(request, 'dashboard/authors_list.html', context)
    except Exception as e:
        messages.error(request, f"Error fetching authors: {str(e)}")
        return render(request, 'dashboard/authors_list.html', {'error': str(e)})


@login_required(login_url='users:login')
def author_detail(request, author_id):
    try:
        api = get_api_client(request)

        author = api.get_author(author_id)
        publications = api.get_author_publications(author_id)
        recommendations = api.get_author_recommendations(author_id)

        context = {
            'author': author,
            'publications': publications.get('results', []),
            'recommendations': recommendations.get('results', []),
        }
        return render(request, 'dashboard/author_detail.html', context)
    except Exception as e:
        messages.error(request, f"Error fetching author details: {str(e)}")
        return redirect('dashboard:authors_list')


@login_required(login_url='users:login')
def search(request):
    try:
        api = get_api_client(request)
        query = request.GET.get('q', '')

        if not query:
            return redirect('dashboard:authors_list')

        data = api.get_authors(search=query)

        context = {
            'query': query,
            'authors': data.get('results', []),
        }
        return render(request, 'dashboard/search.html', context)
    except Exception as e:
        messages.error(request, f"Search error: {str(e)}")
        return render(request, 'dashboard/search.html', {'error': str(e)})


@login_required(login_url='users:login')
def recommendations(request, author_id):
    try:
        api = get_api_client(request)

        author = api.get_author(author_id)
        recommendations = api.get_author_recommendations(author_id)

        context = {
            'author': author,
            'recommendations': recommendations.get('results', []),
        }
        return render(request, 'dashboard/recommendations.html', context)
    except Exception as e:
        messages.error(request, f"Error fetching recommendations: {str(e)}")
        return redirect('dashboard:author_detail', author_id=author_id)
