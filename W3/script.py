import random

def hit_or_miss_hypersphere(radius, num_points, dimensions):
    inside_hypersphere = 0
    
    for _ in range(num_points):
        point = [random.uniform(-radius, radius) for _ in range(dimensions)]
        distance_squared = sum(coord ** 2 for coord in point)
        
        if distance_squared <= radius**2:
            inside_hypersphere += 1
    
    return (inside_hypersphere / num_points) * ((2 * radius) ** dimensions)

def main():
    radius = float(input("Inserisci il raggio dell'ipersfera: "))
    dimensions = int(input("Inserisci il numero di dimensioni: "))
    num_points = int(input("Inserisci il numero di punti da generare: "))

    estimated_volume = hit_or_miss_hypersphere(radius, num_points, dimensions)
    print("Il volume stimato dell'ipersfera in", dimensions, "dimensioni è:", estimated_volume)

if __name__ == "__main__":
    main()