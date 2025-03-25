import numpy as np
import sys
import io
from typing import Dict, List, Tuple

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class ElasticNetSOM:
    def __init__(self, cities: np.ndarray, distances: Dict[Tuple[str, str], float], city_names: List[str]):
        self.cities = cities
        self.distances = distances
        self.city_names = city_names
        self.num_neurons = int(len(cities) * 1.5)
        self.initial_learning_rate = 0.5
        self.initial_radius = self.num_neurons * 0.5
        self.max_iterations = 1000
        self._initialize_neurons()
    
    def _initialize_neurons(self) -> None:
        centroid = np.mean(self.cities, axis=0)
        max_radius = np.max(np.linalg.norm(self.cities - centroid, axis=1))
        angles = np.linspace(0, 2 * np.pi, self.num_neurons, endpoint=False)
        self.neuron_positions = np.array([
            centroid + max_radius * np.array([np.cos(angle), np.sin(angle)])
            for angle in angles
        ])
    
    def _distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.linalg.norm(point1 - point2)
    
    def _get_distance_between_cities(self, city1_idx: int, city2_idx: int) -> float:
        city1_name = self.city_names[city1_idx]
        city2_name = self.city_names[city2_idx]
        if (city1_name, city2_name) in self.distances:
            return self.distances[(city1_name, city2_name)]
        elif (city2_name, city1_name) in self.distances:
            return self.distances[(city2_name, city1_name)]
        else:
            return self._distance(self.cities[city1_idx], self.cities[city2_idx])
    
    def _influence(self, distance: float, radius: float) -> float:
        return np.exp(-distance**2 / (2 * radius**2))
    
    def train(self) -> None:
        for iteration in range(self.max_iterations):
            learning_rate = self.initial_learning_rate / (1 + iteration * 0.01)
            radius = max(1.0, self.initial_radius / (1 + iteration * 0.01))
            city_idx = np.random.randint(len(self.cities))
            city = self.cities[city_idx]
            distances_to_neurons = [self._distance(city, neuron) for neuron in self.neuron_positions]
            winner_idx = np.argmin(distances_to_neurons)
            for i, neuron in enumerate(self.neuron_positions):
                dist_to_winner = min(abs(i - winner_idx), self.num_neurons - abs(i - winner_idx))
                influence = self._influence(dist_to_winner, radius)
                self.neuron_positions[i] += learning_rate * influence * (city - neuron)
    
    def extract_route(self) -> List[int]:
        neuron_to_city = []
        for neuron in self.neuron_positions:
            distances = [self._distance(neuron, city) for city in self.cities]
            closest_city_idx = np.argmin(distances)
            neuron_to_city.append(closest_city_idx)
        seen = set()
        unique_route = []
        for city_idx in neuron_to_city:
            if city_idx not in seen:
                seen.add(city_idx)
                unique_route.append(city_idx)
        if 0 not in unique_route:
            unique_route.insert(0, 0)
        else:
            start_idx = unique_route.index(0)
            unique_route = unique_route[start_idx:] + unique_route[:start_idx]
        unique_route.append(0)
        return unique_route
    
    def calculate_total_distance(self, route: List[int]) -> float:
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += self._get_distance_between_cities(route[i], route[i+1])
        return total_distance

def main():
    cities = np.array([
        [0, 0],    # C1 (Start)
        [12, 12],  # C2
        [10, 8],   # C3
        [20, 8],   # C4
        [15, 3],   # C5
        [25, 0],   # C6
        [5, 5]     # C7
    ])
    
    distances = {
        ('1', '2'): 12, ('1', '3'): 10, ('1', '7'): 12,
        ('2', '3'): 8,  ('2', '4'): 12,
        ('3', '4'): 11, ('3', '5'): 3, ('3', '7'): 9,
        ('4', '5'): 11, ('4', '6'): 10,
        ('5', '6'): 6,  ('5', '7'): 7,
        ('6', '7'): 9
    }
    
    city_names = ["1", "2", "3", "4", "5", "6", "7"]
    som = ElasticNetSOM(cities, distances, city_names)
    som.train()
    optimal_route = som.extract_route()
    total_distance = som.calculate_total_distance(optimal_route)
    
    print("Optimal Route:")
    print(" → ".join([city_names[idx] for idx in optimal_route]))
    print(f"Total Distance: {total_distance:.2f}")

if __name__ == "__main__":
    main()
