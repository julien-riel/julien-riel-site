---
title: "Les bonnes pratiques Python en 2024"
date: 2024-02-01
tags:
  - python
  - developpement
  - bonnes-pratiques
description: "Guide des meilleures pratiques pour ecrire du code Python propre, maintenable et performant."
---

Python est un langage versatile, mais ecrire du bon code Python demande de suivre certaines conventions.

## Structure du projet

### Organisation des fichiers

Un projet Python bien structure facilite la maintenance :

```
mon-projet/
├── src/
│   └── mon_package/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_core.py
├── pyproject.toml
└── README.md
```

### Configuration moderne avec pyproject.toml

```toml
[project]
name = "mon-projet"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "requests>=2.28",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.1",
]
```

## Typage et validation

### Type hints

Le typage ameliore la lisibilite et permet la detection d'erreurs :

```python
from typing import Optional
from dataclasses import dataclass

@dataclass
class User:
    name: str
    email: str
    age: Optional[int] = None

    def is_adult(self) -> bool:
        return self.age is not None and self.age >= 18
```

### Validation avec Pydantic

```python
from pydantic import BaseModel, EmailStr, field_validator

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    age: int

    @field_validator("age")
    @classmethod
    def age_must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("L'age doit etre positif")
        return v
```

## Tests automatises

### Structure des tests

```python
import pytest
from mon_package.core import User

class TestUser:
    def test_create_user(self):
        user = User(name="Alice", email="alice@example.com")
        assert user.name == "Alice"

    def test_is_adult_with_age(self):
        user = User(name="Bob", email="bob@example.com", age=25)
        assert user.is_adult() is True

    @pytest.mark.parametrize("age,expected", [
        (17, False),
        (18, True),
        (None, False),
    ])
    def test_is_adult_edge_cases(self, age, expected):
        user = User(name="Test", email="test@example.com", age=age)
        assert user.is_adult() is expected
```

## Conclusion

Ces pratiques vous aideront a ecrire du code Python de qualite professionnelle.
