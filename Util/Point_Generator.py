import json

points = []
point = 1

for i in range(16):
    for j in range(16):
        points.append({
            "point": point,
            "X": i * 9,
            "Y": j * 9
        })
        point += 1

data = {"points": points}

# Save to a .json file
with open("../Points/Base_Point_grid.json", "w") as f:
    json.dump(data, f, indent=3)

