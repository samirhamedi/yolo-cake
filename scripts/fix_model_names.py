from ultralytics import YOLO

CORRECT_NAMES = {
    0: "Chocolate Cake",
    1: "Chocolate Coated",
    2: "Jam Coated",
    3: "Red Velvet",
    4: "Sponge Cake",
    5: "Vermicelli",
    6: "Whipped Cream",
}

model = YOLO("history/best.pt")
print("Before:", model.names)
model.model.names = CORRECT_NAMES
model.save("history/best-fixed.pt")

check = YOLO("history/best-fixed.pt")
print("After :", check.names)
