import requests
from django.conf import settings
from decouple import config


class FastAPIClient:
    def __init__(self, base_url=None):
        self.base_url = base_url or config('API_BASE_URL', default='http://localhost:8001/')
        self.session = requests.Session()

    def _request(self, method, endpoint, **kwargs):
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API Error: {str(e)}")

    # ==================== FOLKS ====================
    def get_all_people(self):
        return self._request('GET', 'folks')

    def get_person_by_name(self, name):
        return self._request('GET', f'folks/{name}')

    def get_person_by_id(self, person_id):
        return self._request('GET', f'folks/id/{person_id}')

    def create_person(self, name, data):
        return self._request('POST', f'folks/{name}', json=data)

    def update_person(self, name, data):
        return self._request('PUT', f'folks/{name}', json=data)

    def delete_person(self, name):
        return self._request('DELETE', f'folks/{name}')

    # ==================== USERS ====================
    def register_user(self, username, password, folk_name, is_admin=False):
        data = {
            "username": username,
            "password": password,
            "folk_name": folk_name,
            "is_admin": is_admin
        }
        return self._request('POST', 'users/register', json=data)

    def get_all_users(self):
        return self._request('GET', 'users')

    def get_user(self, user_id):
        return self._request('GET', f'users/{user_id}')

    def get_user_folk(self, user_id):
        return self._request('GET', f'users/{user_id}/folk')

    def update_user(self, user_id, data):
        return self._request('PUT', f'users/{user_id}', json=data)

    # ==================== WORKS ====================
    def get_all_works(self):
        return self._request('GET', 'works')

    def get_work(self, work_id):
        return self._request('GET', f'works/{work_id}')

    def create_work(self, title, authors, year, citations, doi=None, url=None):
        data = {
            "title": title,
            "authors": authors,
            "year": year,
            "citations": citations,
            "doi": doi,
            "url": url
        }
        return self._request('POST', 'works', json=data)

    def get_works_by_author(self, person_id):
        return self._request('GET', f'works/by-author/{person_id}')

    def health_check(self):
        return self._request('GET', 'health')
