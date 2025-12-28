---
title: "Les bonnes pratiques Python en 2024"
date: 2024-02-01
tags:
  - python
  - developpement
  - bonnes-pratiques
description: "Guide des meilleures pratiques pour écrire du code Python propre, maintenable et performant."
---

Python est un langage versatile, mais écrire du bon code Python demande de suivre certaines conventions.

<img src="/assets/images/code-illustration.webp" alt="Illustration de code" width="400" height="267" loading="lazy" decoding="async">

## Structure du projet

### Organisation des fichiers

Un projet Python bien structuré facilite la maintenance :

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

Le typage améliore la lisibilité et permet la détection d'erreurs :

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
            raise ValueError("L'âge doit être positif")
        return v
```

## Tests automatisés

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

Ces pratiques vous aideront à écrire du code Python de qualité professionnelle.
