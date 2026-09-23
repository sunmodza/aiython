from dataclasses import dataclass
from pathlib import Path

ASSETS = Path(__file__).resolve().parent

@dataclass
class Product:
    name: str
    image_path: str

products = [Product("Red shoe", str(ASSETS / "red-shoe.jpg")), Product("Blue boot", str(ASSETS / "blue-boot.jpg"))]
# A second camera angle of the catalog's red running shoe.
query_image_path = str(ASSETS / "shoe.jpg")

results: list[Product] = compare the image files in products with query_image_path and return the original best matching Product in a one-item list
assert all(any(item is original for original in products) for item in results)
print([item.name for item in results])
