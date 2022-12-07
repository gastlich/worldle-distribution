from typing import Optional
from pydantic import BaseModel


class Country(BaseModel):
    name: str
    code: Optional[str]
    latitude: float
    longitude: float

    class Config:
        orm_mode = True
