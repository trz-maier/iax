import time
import numpy as np
import matplotlib.pyplot as plt
from random import random
from iax import utilities, DefaultCostFunction
from typing import List
from IPython.core.display import clear_output
from iax.distance import EuclideanDistance
from iax.inferfaces import Engine, Input, CostFunction


class Particle:
    def __init__(self, position, bounds):
        self.position = position
        self.velocity = 2 * np.random.rand(len(position)) - 1
        self.bounds = bounds
        self.best_position = position
        self.best_value = np.inf
        self.best_local_position = position
        self.best_local_value = np.inf
        self.neighbors = []

    def __str__(self):
        return "Personal best: %s" % self.best_value

    def update_position(self):
        self.position = np.add(self.position, self.velocity)
        for idx in range(len(self.position)):
            if self.position[idx] < self.bounds[idx][0]:
                self.position[idx] = self.bounds[idx][0]
            if self.position[idx] > self.bounds[idx][1]:
                self.position[idx] = self.bounds[idx][1]


class ParticleCollection:
    def __init__(self, num_particles, starting_position, bounds):
        self.particle_list = []
        self.best_position = None
        self.best_value = np.inf
        self._bounds = bounds
        self._num_particles = num_particles
        self._starting_position = starting_position

        self._generate_particles()
        self._assign_neighbors()

    def _generate_particles(self):
        for _ in range(self._num_particles):
            self.particle_list.append(Particle(self._starting_position, self._bounds))

    def _assign_neighbors(self):
        for p1 in self.particle_list:
            for p2 in self.particle_list:
                if p1 != p2:
                    p1.neighbors.append(p2)


class Space:
    def __init__(self, particles: List[Particle]):
        self.particles = particles
        self.global_best_value = np.inf
        self.global_best_position = None

    def __str__(self):
        out = ""
        for particle in self.particles:
            out += str(particle)
            out += "\n"
        return out

    def move_particles(self, w, c1, c2, c3):
        for particle in self.particles:
            for idx in range(len(particle.velocity)):
                x1 = c1 * random() * (particle.best_position[idx] - particle.position[idx])
                x2 = c2 * random() * (particle.best_local_position[idx] - particle.position[idx])
                x3 = c3 * random() * (self.global_best_position[idx] - particle.position[idx])
                particle.velocity[idx] = w * particle.velocity[idx] + x1 + x2 + x3
            particle.update_position()


class PSO(Engine):
    def __init__(self, number_of_particles: List[int], bounds: List[int] = None,
                 cost_function: CostFunction = DefaultCostFunction):

        self._number_of_particles = number_of_particles
        self._x = None
        self._classifier = None
        self._space = None
        self._starting_position = None
        self._shape = None
        self._max_distance = np.inf
        self._bounds = bounds
        self._cost_function = cost_function
        self._distance_function = EuclideanDistance()

        self._out = []
        self.__status = ""
        super().__init__("Adversarial Particle Swarm Optimisation")

    @property
    def input(self):
        return self._x

    @property
    def max_distance(self):
        return self._max_distance

    @property
    def output(self):
        return self._out

    def initialize(self, x: Input, classifier):

        self._x = x
        self._classifier = classifier

        if self._x.mask is None:
            masked = self._x.input
        else:
            masked = self._x.input[self._x.mask]

        self._starting_position = masked.flatten()

        bounds = np.array([self._bounds] * self._starting_position.shape[0])
        col_list = [ParticleCollection(x, self._starting_position, bounds) for x in self._number_of_particles]

        particle_list = []
        for list_ in col_list:
            for p_ in list_.particle_list:
                particle_list.append(p_)

        self._shape = masked.shape
        self._space = Space(particle_list)

    def search(self, iterations: int, w: float, c1: float, c2: float, c3: float,
               terminate: bool = False, max_distance: int = np.inf):

        start = time.time()

        number_of_particles = len(self._space.particles)
        self._max_distance = max_distance

        def print_progress(i, idx):
            clear_output(wait=True)
            n = number_of_particles
            print("Iteration : %s/%s %s\nStatus    : %s\nBest      : %s\nFound     : %s"
                  % (i, iterations, utilities.progress_bar(idx, n * 3 + 1), self.__status,
                     self._space.global_best_value, found))

        found = 0
        best_distance = np.inf
        iteration = 0
        index = 0

        while iteration < iterations:

            iteration += 1
            index = 0

            # evaluate, update personal best
            for p in self._space.particles:
                candidate = self.input.update(p.position.reshape(self._shape))
                distance = self.input.get_distance(self._distance_function)
                cost, confidence, adversarial = self._cost_function.calculate(self._classifier, candidate, self.input.label)

                if distance > self._max_distance:
                    candidate_fitness = np.inf
                else:
                    candidate_fitness = cost

                d = {'value': candidate, 'adversarial': adversarial, 'distance': distance,
                     'confidence': confidence, 'iteration': iteration}
                self._out.append(d)

                if adversarial and distance < self._max_distance:
                    found += 1
                    if distance < best_distance:
                        best_distance = distance

                if p.best_value > candidate_fitness:
                    p.best_value = candidate_fitness
                    p.best_position = p.position

                self.__status = "Searching at distance %s" % int(distance)
                index += 1
                print_progress(iteration, index)

            # update local best
            self.__status = "Updating local best"
            for p in self._space.particles:
                for n in p.neighbors:
                    if n.best_value < p.best_local_value:
                        p.best_local_value = n.best_value
                        p.best_local_position = n.best_position
                    else:
                        p.best_local_value = p.best_value
                        p.best_local_position = p.best_position

                print_progress(iteration, index)
                index += 1

            # update global best
            self.__status = "Updating global best"
            for p in self._space.particles:
                if self._space.global_best_value > p.best_value:
                    self._space.global_best_value = p.best_value
                    self._space.global_best_position = p.best_position

                index += 1
                print_progress(iteration, index)

            self.__status = "Moving particles"
            print_progress(iteration, index)
            self._space.move_particles(w, c1, c2, c3)
            index += 1
            print_progress(iteration, index)

            # if adversarial example found terminate while loop
            if terminate and found > 0:
                break

        self.__status = "Completed search in %s" % utilities.format_seconds(int(time.time() - start))
        print_progress(iteration, index)

    def get_adversarial_examples(self, sort_by: str = None):
        ax = self.output.copy()
        if sort_by:
            ax.sort(key=lambda x: x[sort_by], reverse=False)
        return list(filter(lambda x: x['adversarial'] and x['distance'] < self._max_distance, ax))

    def plot(self):

        plt.figure(figsize=[15, 7])
        x1 = [[x['distance'], x['confidence']] for x in filter(lambda x: not x['adversarial'], self.output)]
        x2 = [[x['distance'], x['confidence']] for x in filter(lambda x: x['adversarial'], self.output)]
        plt.scatter([x[0] for x in x1], [x[1] for x in x1], s=2, alpha=0.5, label='non-adversarial')
        plt.scatter([x[0] for x in x2], [x[1] for x in x2], s=2, alpha=0.5, c='red', label='adversarial')
        plt.vlines(self.max_distance, 0, 1, linewidth=0.5, label='max distance')
        plt.xlabel('distance')
        plt.ylabel('confidence')
        plt.legend()

        plt.figure(figsize=[15, 7])
        x1 = [x['distance'] if not x['adversarial'] else None for x in self.output]
        x2 = [x['distance'] if x['adversarial'] else None for x in self.output]
        plt.scatter(range(len(x1)), x1, s=2, alpha=0.5, label='non-adversarial')
        plt.scatter(range(len(x2)), x2, s=2, alpha=0.5, c='red', label='adversarial')
        plt.hlines(self.max_distance, 0, len(x1), linewidth=0.5, label='max distance')
        plt.xlabel('query')
        plt.ylabel('distance')
        plt.legend()
