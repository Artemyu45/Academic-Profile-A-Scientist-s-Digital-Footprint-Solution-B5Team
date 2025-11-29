from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import json
import os
from pathlib import Path

# Initialize FastAPI
app = FastAPI(
    title="Academic Profile API",
    description="REST API for Academic Profile System",
    version="1.0.0"
)

# CORS middleware to allow Django frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database path
DB_DIR = Path(__file__).parent / "db"
FOLKS_DB = DB_DIR / "folks.json"
USERS_DB = DB_DIR / "users.json"
WORKS_DB = DB_DIR / "works.json"

# Ensure db directory exists
DB_DIR.mkdir(exist_ok=True)


# ==================== PYDANTIC MODELS ====================

class Neighbor(BaseModel):
    id: int


class Publication(BaseModel):
    links: str


class Person(BaseModel):
    id: int
    citation_impact: int
    publication_count: int
    research_field: int
    nearest_neighbors: List[Neighbor] = []
    publication: List[Publication] = []


class User(BaseModel):
    username: str
    password: str
    is_admin: bool
    folk_name: str


class Work(BaseModel):
    id: int
    title: str
    authors: List[str]
    year: int
    citations: int
    doi: str = None
    url: str = None


# ==================== DATABASE FUNCTIONS ====================

def load_json(filepath: Path) -> Dict:
    """Load JSON file or return empty dict"""
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_json(filepath: Path, data: Dict):
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_persons() -> Dict:
    return load_json(FOLKS_DB)


def load_users() -> Dict:
    data = load_json(USERS_DB)
    # Convert string keys to int if needed
    return {int(k) if k.isdigit() else k: v for k, v in data.items()}


def load_works() -> Dict:
    data = load_json(WORKS_DB)
    return {int(k) if k.isdigit() else k: v for k, v in data.items()}


def save_persons(data: Dict):
    save_json(FOLKS_DB, data)


def save_users(data: Dict):
    save_json(USERS_DB, {str(k): v for k, v in data.items()})


def save_works(data: Dict):
    save_json(WORKS_DB, {str(k): v for k, v in data.items()})


# ==================== HELPER FUNCTIONS ====================

def find_persons_by_id(person_id: int) -> tuple:
    """Find person by ID"""
    persons = load_persons()
    for name, data in persons.items():
        if data.get("id") == person_id:
            return name, data
    return None, None


def find_next_id() -> int:
    """Find next available ID"""
    persons = load_persons()
    users = load_users()
    works = load_works()

    max_person_id = max([data["id"] for data in persons.values()], default=0)
    max_user_id = max(users.keys() if isinstance(list(users.keys())[0], int) else [int(k) for k in users.keys()],
                      default=0)
    max_work_id = max(works.keys() if works and isinstance(list(works.keys())[0], int) else [0], default=0)

    return max(max_person_id, max_user_id, max_work_id) + 1


# ==================== FOLKS ENDPOINTS ====================

@app.get("/folks", summary="Get all people", tags=["Folks Management"])
def get_all_people():
    """Get all people in the system"""
    return load_persons()


@app.get("/folks/{full_name}", summary="Get person by name", tags=["Folks Management"])
def get_person(full_name: str):
    """Get a specific person by full name"""
    persons = load_persons()
    if full_name not in persons:
        raise HTTPException(status_code=404, detail="Person not found")
    return {full_name: persons[full_name]}


@app.get("/folks/id/{person_id}", summary="Get person by ID", tags=["Folks Management"])
def get_person_by_id(person_id: int):
    """Get a specific person by ID"""
    name, data = find_persons_by_id(person_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return {name: data}


@app.post("/folks/{full_name}", summary="Create new person", tags=["Folks Management"])
def create_person(full_name: str, person: Person):
    """Create a new person"""
    persons = load_persons()
    if full_name in persons:
        raise HTTPException(status_code=400, detail="Person already exists")

    persons[full_name] = person.dict()
    save_persons(persons)
    return {full_name: persons[full_name]}


@app.put("/folks/{full_name}", summary="Update person", tags=["Folks Management"])
def update_person(full_name: str, person: Person):
    """Update an existing person"""
    persons = load_persons()
    if full_name not in persons:
        raise HTTPException(status_code=404, detail="Person not found")

    persons[full_name] = person.dict()
    save_persons(persons)
    return {full_name: persons[full_name]}


@app.put("/folks/id/{person_id}", summary="Update person by ID", tags=["Folks Management"])
def update_person_by_id(person_id: int, data: dict = Body(...)):
    """Update person data by ID"""
    persons = load_persons()
    for fullname in persons:
        if persons[fullname]["id"] == person_id:
            persons[fullname].update(data)
            save_persons(persons)
            return {fullname: persons[fullname]}

    raise HTTPException(status_code=404, detail="Person not found")


@app.delete("/folks/{full_name}", summary="Delete person", tags=["Folks Management"])
def delete_person(full_name: str):
    """Delete a person by name"""
    persons = load_persons()
    if full_name not in persons:
        raise HTTPException(status_code=404, detail="Person not found")

    del persons[full_name]
    save_persons(persons)
    return {"message": f"{full_name} deleted"}


@app.delete("/folks/id/{person_id}", summary="Delete person by ID", tags=["Folks Management"])
def delete_person_by_id(person_id: int):
    """Delete a person by ID"""
    persons = load_persons()
    for fullname in list(persons.keys()):
        if persons[fullname]["id"] == person_id:
            del persons[fullname]
            save_persons(persons)
            return {"message": f"Person with id {person_id} deleted"}

    raise HTTPException(status_code=404, detail="Person not found")


# ==================== USERS ENDPOINTS ====================



@app.post("/users/register", summary="Register new user", tags=["Users Management"])
def register_user(user: User):
    """Register a new user"""
    users = load_users()
    persons = load_persons()

    # Check if username already exists
    if any(u["username"] == user.username for u in users.values()):
        raise HTTPException(status_code=400, detail="Username already exists")

    # Find or create folk
    if user.folk_name in persons:
        folk_id = persons[user.folk_name]["id"]
    else:
        new_id = find_next_id()
        folk_id = new_id
        persons[user.folk_name] = {
            "id": folk_id,
            "citation_impact": 0,
            "publication_count": 0,
            "research_field": 0,
            "nearest_neighbors": [],
            "publication": []
        }
        save_persons(persons)

    # Create user
    new_user_id = folk_id
    users[new_user_id] = {
        "id": new_user_id,
        "username": user.username,
        "password": user.password,  # TODO: Hash password in production
        "is_admin": user.is_admin,
        "folk_name": user.folk_name
    }
    save_users(users)

    return {
        "user": users[new_user_id],
        "folk": persons[user.folk_name]
    }


@app.get("/users", summary="Get all users", tags=["Users Management"])
def get_all_users():
    """Get all users"""
    return load_users()


@app.get("/users/{user_id}", summary="Get user by ID", tags=["Users Management"])
def get_user(user_id: int):
    """Get a specific user"""
    users = load_users()
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    return users[user_id]


@app.get("/users/{user_id}/folk", summary="Get user's associated folk", tags=["Users Management"])
def get_user_folk(user_id: int):
    """Get the folk associated with a user"""
    users = load_users()
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")

    persons = load_persons()
    folk_name = users[user_id]["folk_name"]

    if folk_name not in persons:
        raise HTTPException(status_code=404, detail="Associated folk not found")

    return {folk_name: persons[folk_name]}


@app.put("/users/{user_id}", summary="Update user", tags=["Users Management"])
def update_user(user_id: int, updated_data: dict = Body(...)):
    """Update user data"""
    users = load_users()
    persons = load_persons()

    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")

    user = users[user_id]

    # Update allowed fields
    for key in ["username", "password", "is_admin", "folk_name"]:
        if key in updated_data:
            user[key] = updated_data[key]

    # Create folk if doesn't exist
    new_folk_name = updated_data.get("folk_name")
    if new_folk_name and new_folk_name not in persons:
        persons[new_folk_name] = {
            "id": user_id,
            "citation_impact": 0,
            "publication_count": 0,
            "research_field": 0,
            "nearest_neighbors": [],
            "publication": []
        }
        save_persons(persons)

    save_users(users)

    return {
        "user": user,
        "folk": persons.get(user["folk_name"])
    }


@app.delete("/users/{user_id}", summary="Delete user", tags=["Users Management"])
def delete_user(user_id: int):
    """Delete a user"""
    users = load_users()
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")

    del users[user_id]
    save_users(users)

    return {"message": f"User {user_id} deleted"}

import csv
from fastapi import APIRouter, HTTPException
from main import register_user, load_users, load_persons, save_users, save_persons, User


router = APIRouter()

@app.post("/users/register_csv", tags=["Users Management"])
def register_users_from_csv(csv_path: str):
    """
    Register multiple users from a CSV file.
    CSV columns: username, password, is_admin, folk_name
    """
    users = load_users()
    persons = load_persons()

    created = []
    errors = []

    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    user = User(
                        username=row["username"],
                        password=row["password"],
                        is_admin=row.get("is_admin", "false").lower() == "true",
                        folk_name=row["folk_name"],
                    )
                    # --- Reuse logic from register_user ---
                    if any(u["username"] == user.username for u in users.values()):
                        raise HTTPException(status_code=400, detail="Username already exists")

                    # Folk handling
                    if user.folk_name in persons:
                        folk_id = persons[user.folk_name]["id"]
                    else:
                        # Create new folk
                        new_id = max(int(k) for k in users.keys()) + 1 if users else 1
                        folk_id = new_id
                        persons[user.folk_name] = {
                            "id": folk_id,
                            "citation_impact": 0,
                            "publication_count": 0,
                            "research_field": 0,
                            "nearest_neighbors": [],
                            "publication": []
                        }

                    # Create user
                    users[folk_id] = {
                        "id": folk_id,
                        "username": user.username,
                        "password": user.password,  # TODO: Replace with hashing
                        "is_admin": user.is_admin,
                        "folk_name": user.folk_name
                    }

                    created.append(user.username)
                except Exception as e:
                    errors.append({"row": row, "error": str(e)})

        save_users(users)
        save_persons(persons)

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="CSV file not found")

    return {"created": created, "errors": errors}


# ==================== WORKS ENDPOINTS ====================

@app.get("/works", summary="Get all works", tags=["Works Management"])
def get_all_works():
    """Get all works/publications"""
    return load_works()


@app.get("/works/{work_id}", summary="Get work by ID", tags=["Works Management"])
def get_work(work_id: int):
    """Get a specific work"""
    works = load_works()
    if work_id not in works:
        raise HTTPException(status_code=404, detail="Work not found")
    return {work_id: works[work_id]}


@app.post("/works", summary="Create new work", tags=["Works Management"])
def create_work(work: Work):
    """Create a new work/publication"""
    works = load_works()

    new_id = find_next_id()
    work_dict = work.dict()
    work_dict["id"] = new_id

    works[new_id] = work_dict
    save_works(works)

    return {new_id: works[new_id]}


@app.put("/works/{work_id}", summary="Update work", tags=["Works Management"])
def update_work(work_id: int, work: Work):
    """Update a work/publication"""
    works = load_works()
    if work_id not in works:
        raise HTTPException(status_code=404, detail="Work not found")

    works[work_id] = work.dict()
    save_works(works)

    return {work_id: works[work_id]}


@app.delete("/works/{work_id}", summary="Delete work", tags=["Works Management"])
def delete_work(work_id: int):
    """Delete a work"""
    works = load_works()
    if work_id not in works:
        raise HTTPException(status_code=404, detail="Work not found")

    del works[work_id]
    save_works(works)

    return {"message": f"Work {work_id} deleted"}


@app.get("/works/by-author/{person_id}", summary="Get works by author", tags=["Works Management"])
def get_works_by_author(person_id: int):
    """Get all works by a specific author"""
    works = load_works()
    author_works = {}

    for work_id, work_data in works.items():
        if person_id in work_data.get("author_ids", []):
            author_works[work_id] = work_data

    return author_works


# ==================== HEALTH CHECK ====================

@app.get("/health", summary="Health check", tags=["System"])
def health_check():
    """Check if API is running"""
    return {
        "status": "ok",
        "message": "Academic Profile API is running"
    }


# ==================== ROOT ====================

@app.get("/", summary="API Info", tags=["System"])
def root():
    """Root endpoint with API information"""
    return {
        "name": "Academic Profile API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)