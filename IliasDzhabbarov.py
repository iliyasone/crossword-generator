from __future__ import annotations
from dataclasses import dataclass
import os

from random import randint, seed, choice, choices, sample, shuffle
from typing import NamedTuple
from itertools import product

from functools import lru_cache

from copy import deepcopy
N = 20
POPULATION_NUMBER = 100

HORIZONTAL = 0
VERTICAL = 1

def load_chromosome_from_file(path = 'test.txt'):
    chromosome = Chromosome()
    with open(path, 'r') as file:
        for line, word in zip(file.readlines(), chromosome.words):
            row, col, ori = map(int, line.strip().split())
            word.row = row
            word.column = col
            word.orientation = ori
    return chromosome



def generate_crossword(words) -> list[int]:
    #seed(100)
    Chromosome.WORDS = words
        
    population = generate_population()
    
    # chromosome = load_chromosome_from_file()
    # print_chromosome(chromosome)
    # print(fitness(chromosome))
    # input("CONTINUE: ")
    
    best_fitness = -10
    with_fitness = [(fitness(chromosome), chromosome) for chromosome in population]
    
    avg_fitness_history = []
    i = 0
    while best_fitness != 0:
        i += 1
        print('-'*50)
        print(f"GENERATION {i}")
        
        
        
        noize = None
        if len(avg_fitness_history) > 5:
            if abs(sum(avg_fitness_history[-5:]) / 5 - best_fitness) < 0.1: 
                noize = True
        with_fitness.sort(key= lambda x: x[0],reverse=True)
        with_fitness = evolution_step(with_fitness=with_fitness, noise = noize)
        
        
        

        avg_fitness = sum(x[0] for x in with_fitness) / len(with_fitness)
        avg_fitness_history.append(avg_fitness)
        print(f"AVG FITNESS {avg_fitness}, FOR POPULATION: {len(with_fitness)}")
        with_fitness.sort(key= lambda x: x[0],reverse=True)
        
        for fitness_v, chromosome in with_fitness[:1]:
            print('fitness:', fitness_v, fitness(chromosome))
            print_chromosome(chromosome)
        
        best_fitness = with_fitness[0][0]
        
        # if i == 100:
        #     input("PRESS ENTER TO PRINT ALL POPULATION")
        #     for fitness_v, chromosome in with_fitness:
        #         print('-'*20)
        #         print('fitness:', fitness_v)
        #         print_chromosome(chromosome)
        #     input("CONTINUE?")
    
    #print(*with_fitness[:5],sep='\n')
    
    return chromosome_to_list(with_fitness[0][1])


def select_pair(with_fitness: list[tuple[int, Chromosome]]) -> list[tuple[int, Chromosome]]:
    min_fitness = min(x[0] for x in with_fitness)
    b = -min_fitness + 1
    
    r1 = 0
    r2 = 0
    
    while r1 == r2:
        r1, r2 = choices(
        population=with_fitness,
        weights=tuple((b+x[0])**0.5 for x in with_fitness),
        k=2
        )    
    
    return r1,r2 

def evolution_step(population: list[Chromosome] = None, with_fitness: list[tuple[int, Chromosome]] = None):
    """Rules for selection:
    
    Keep 2 top

    Mutate 10% population
    
    Crossover all
    """
    
    ELTITISM = 25
    CROSSOVER = 80
    MUTATION = 3
    
    if with_fitness is None:
        with_fitness = [(fitness(chromosome), chromosome) for chromosome in population]
        with_fitness.sort(key= lambda x: x[0],reverse=True)
        
    
    if with_fitness[0][0] == 0:
        return [with_fitness[0][1]]
    #new_population = [x[1] for x in with_fitness[:ELTITISM]]
    
    new_with_fitness = with_fitness.copy()[:ELTITISM ]
    
    
    # crossover 20 best
    
    crossover_list  = with_fitness[:CROSSOVER]
    for _ in range(CROSSOVER//2):
        a, b = select_pair(crossover_list)
        crossover_list.remove(a)
        crossover_list.remove(b)
        for offspring in crossover(a[1], b[1]):
            new_with_fitness.append((fitness(offspring),offspring))
            
       # mutation of population
    for x in choices(with_fitness, 
                      k=MUTATION):
        new_chomosome = mutation(x[1])
        if randint(0,1) == 0:
            new_chomosome = mutation(new_chomosome)
        new_with_fitness.append((fitness(new_chomosome), new_chomosome))
    
    new_with_fitness = list(set(new_with_fitness))
    new_with_fitness.sort(key= lambda x: x[0],reverse=True)
    return new_with_fitness[:POPULATION_NUMBER]


def evolution_step(with_fitness: list[tuple[int, Chromosome]], noise = None):
    """Crossover
    
    Mutation
    
    Selection
    """
    CROSSOVER = 90
    MUTATION = 18
    SELECTION_NUMBER = 2
    NOISE = 50
    crossover_population: list[Chromosome] = []

    if noise:
        with_fitness.sort(key=lambda x: -x[0])
        print("NOISE!")
        noise_chromosomes = [Chromosome() for _ in range(NOISE)]
        noise_with_fitness = [(fitness(chromo), chromo) for chromo in noise_chromosomes]
        with_fitness = with_fitness[:POPULATION_NUMBER-NOISE] + noise_with_fitness
        
    crossover_population = [x[1] for x in with_fitness][:POPULATION_NUMBER-CROSSOVER]
    shuffle(with_fitness)
    max_fitness = max(-x[0] for x in with_fitness)

    
    for _ in range(CROSSOVER//2):
        weights = tuple(max_fitness*1.5+ x[0] for x in with_fitness)
        a, b = choices(
            with_fitness,
            weights=weights,
            k=2)
        
        a, b = crossover(a[1], b[1])
        crossover_population.append(a)
        crossover_population.append(b)
    
    for chromosome in choices(
        crossover_population,
        k=MUTATION
        ):
        # i = randint(0, len(Chromosome.WORDS)-1)
        # new_word = Chromosome().words[i]
        
        # chromosome.words[i] = new_word
        mutate(chromosome)
        if randint(0,1) == 0:
            mutate(chromosome)
            if randint(0,1) == 0:
                mutate(chromosome)
                if randint(0,1) == 0:
                     mutate(chromosome)
    
    crossover_population = [(fitness(chros), chros) for chros in crossover_population]
    new_population = []
    
    while len(new_population) != POPULATION_NUMBER:
        group = choices(
            crossover_population,
            k=SELECTION_NUMBER
        )
        
        
        group.sort(key=lambda x: x[0])
        #crossover_population.remove(group[-1])
        new_population.append(group[-1])
    

    
    return new_population        
            

        
        
        
    
    
    
    
    

def chromosome_to_list(chromosome: Chromosome) -> list[list[int]]:
    result = []
    for word in chromosome.words:
        result.append([word.row, word.column, word.orientation])
    return result


def generate_population(n: int = POPULATION_NUMBER) -> list[Chromosome]:
    return [Chromosome() for _ in range(n)]


def mutate(chromosome: Chromosome) -> None:
    mutation(chromosome, in_place=True)
    
MUTATION_STATISTICS = {"COLUMN": 0, "ROW": 0, "ORIENTATION": 0}

def mutation(chromosome: Chromosome, in_place: bool = False) -> Chromosome:
    if not in_place:
        mutated_chromosome = chromosome.copy()
    else:
        mutated_chromosome = chromosome
    
    word_to_change = randint(0, len(mutated_chromosome.words)-1)
    
    mutated_word = mutated_chromosome.words[word_to_change]
    
    choose, = choices(
        population=[0,1,2],
        weights=[2,2,1],
        k=1
    )
    
    #print(f'{mutated_word=}')
    if choose == 0:
        d = choice((1,-1))
        
        mutated_word.column += d
        if mutated_word.is_fit():
            MUTATION_STATISTICS['COLUMN'] += 1
            return mutated_chromosome
        else:
            mutated_word.column -= 2*d
            if mutated_word.is_fit():
                MUTATION_STATISTICS['COLUMN'] += 1
                return mutated_chromosome
            else:
                choose += 1
    if choose == 1:
        d = choice((1,-1))
        mutated_word.row += d
        if mutated_word.is_fit():
            MUTATION_STATISTICS['ROW'] += 1
            return mutated_chromosome
        else:
            mutated_word.row -= 2*d
            if mutated_word.is_fit():
                MUTATION_STATISTICS['ROW'] += 1
                return mutated_chromosome
            else:
                raise ValueError('Something went wrong during generating word')
    elif choose == 2:
        mutated_word.orientation = int(not mutated_word.orientation) 
        
        if mutated_word.is_fit():
            MUTATION_STATISTICS['ORIENTATION'] += 1
            return mutated_chromosome
        row, column = mutated_word.max_position()
        
        mutated_word.row = min(row, mutated_word.row)
        mutated_word.column = min(column, mutated_word.column)

        return mutated_chromosome
    
def crossover(parent1: Chromosome, parent2: Chromosome):
    n = len(Chromosome.WORDS)
    
    chield1 = Chromosome(parent1)
    chield2 = Chromosome(parent2)
    
    for i in range(n-1):
        if randint(0,1) == 0:
            continue
        chield1.words[i], chield2.words[i] = chield2.words[i], chield1.words[i]
        
    return chield1, chield2


@dataclass
class Word:
    word: str
    row: int = 0
    column: int = 0
    orientation: int = 0
    
    def __len__(self):
        return len(self.word)
    
    def copy(self):
        return Word(word=self.word, row=self.row, column=self.column, orientation=self.orientation)
    
    def max_position(self) -> tuple[int, int]:
        """based on ```word.orientation``` and grid size
        return max word position as ```(row: int, column: int)```"""
        if self.orientation == VERTICAL:
            column = N - 1
            row = N - 1 - len(self)
        elif self.orientation == HORIZONTAL:
            column = N - 1 - len(self)
            row = N - 1
        return row, column
    
    def is_fit(self) -> bool:
        """is word fit to the 20x20 grid"""
        row, column = self.max_position()
        
        if 0 <=  self.column <= column and 0 <= self.row <= row:
            return True
        else:
            return False
class Chromosome:
    WORDS: list[str]
    
    def __init__(self, words: list[Word] | Chromosome | None = None) -> None:
        if words is None:
            self.words = [Word(word) for word in self.__class__.WORDS]
        elif isinstance(words, Chromosome):
            self.words = deepcopy(words.words)
            return
        elif isinstance(words, list):
            self.words = deepcopy(words)
            return
        
        for word in self.words:
            word.orientation = randint(0, 1)
            
            row, column = word.max_position()
            
            word.column = randint(0, column)
            word.row = randint(0, row)
            
                

    
    def __hash__(self) -> int:
        return hash(((word.row, word.column, word.orientation) for word in self.words))
    
    def __eq__(self, other):
        if not isinstance(other, Chromosome):
            return False
        return self.words == other.words
    
    def copy(self):
        return Chromosome(self)


def create_grid(chromosome: Chromosome, empty_cell = None):
    grid = [[empty_cell] * N for _ in range(N)]
    for word in chromosome.words:
        if word.orientation == HORIZONTAL:
            dx = 1
            dy = 0
        elif word.orientation == VERTICAL:
            dx = 0
            dy = 1

        for i in range(len(word)):
            grid[word.row+dy*i][word.column+dx*i] = word.word[i]
    return grid           

def print_chromosome(chromosome: Chromosome):
    for line in create_grid(chromosome, empty_cell='.'):
        print(*line) 
        


class WordPointer(NamedTuple):
    word: Word
    letter: str

def generate_grid_with_word_pointers(chromosome: Chromosome) -> list[list[list[WordPointer]]]:
    """Put in each cell of a grid WordPointer object - NamedTuple that contain
    Word itself (word: str, row: int, columnt: int, orientation: int) and current letter (str)
    
    One cell can contain many WordPointers if words crosses
    """
    grid: list[list[list[WordPointer]]] = [[[] for _ in range(N)] for _ in range(N)]
    
    for word in chromosome.words:
        if word.orientation == HORIZONTAL:
            dx = 1
            dy = 0
        elif word.orientation == VERTICAL:
            dx = 0
            dy = 1

        for i in range(len(word)):
            grid[word.row+dy*i][word.column+dx*i].append(WordPointer(word, word.word[i]))
    return grid

def calculate_point_grade_for_non_equal_letters_and_wrong_orintation(grid: list[list[list[WordPointer]]], x: int, y: int) -> int:
    grade = 0
    intersects = len(grid[y][x])
    if intersects >= 2:
        for i in range(intersects):
            for j in range(i+1, intersects):
                if grid[y][x][i].letter != grid[y][x][j].letter:
                    grade -= 5
                    #print((x,y),grid[y][x][i].letter)
                if grid[y][x][i].word.orientation == grid[y][x][j].word.orientation:
                    grade -= 2
                    #print((x,y),grid[y][x][i].letter)
    return abs(grade)

def calculate_point_grade_checking_neigbours(grid: list[list[list[WordPointer]]], 
                                             x: int, y: int) -> float:
    
    debug = False
    def print(*args):
        pass
    
    intersects = len(grid[y][x])
    grade = 0
    for current_pointer in grid[y][x]:
        for dx, dy in (0,-1),(-1,0),(0,1), (1,0):
            if not(( 0 <= y+dy < N) and (0<= x+dx < N)):
                continue
            
            for neighbor_pointer in grid[y+dy][x+dx]:
                if neighbor_pointer.word == current_pointer.word:
                    continue

                
                # decrease points for each 'gay-connection' (same orintation)
                if neighbor_pointer.word.orientation == current_pointer.word.orientation:
                    is_exception = False
                    
                    # let's check the only exception from this rule
                    # """Exception is first (last) symbol of the first word and last (first)
                    # symbol of the second word, also being parts of the third perpendicular word"""

                    if neighbor_pointer.word.orientation == VERTICAL:
                        min_bound = min(neighbor_pointer.word.row, current_pointer.word.row)
                        
                        max_bound = max(neighbor_pointer.word.row + len(neighbor_pointer.word), 
                                        current_pointer.word.row + len(current_pointer.word)) - 1
                    elif neighbor_pointer.word.orientation == HORIZONTAL:
                        min_bound = min(neighbor_pointer.word.column, current_pointer.word.column)
                        
                        max_bound = max(neighbor_pointer.word.column + len(neighbor_pointer.word), 
                                        current_pointer.word.column + len(current_pointer.word)) - 1
                    
                    ngh = len(neighbor_pointer.word) + len(current_pointer.word) - 2
                    
                    if max_bound - min_bound == len(neighbor_pointer.word) + len(current_pointer.word) - 2:
                    # check if they tangent by the end-or-first letter
                    
                        if intersects == 2 and len(grid[y+dy][x+dx]) == 2:
                            # check if they part of another word   
                                         
                            is_exception = True
                            print((x,y), current_pointer.letter, "BREAK")
                            break
                    if not is_exception:
                        grade -= 2 #/ len(current_pointer.word)
                
                # tangent of 2 perpendicular words)
                elif len(grid[y+dy][x+dx]) == 1 and intersects == 1:
                    print((x,y), current_pointer.letter)
                    print((x+dx, y+dy), neighbor_pointer.letter, 'NEIGBOR')
                    grade -= 1    
                        
        
    return abs(grade)

def get_connected_components_and_their_centers(grid: list[list[list]]):
    visited = [[False] * N for _ in range(N)]
    
    components = []
    current_component = []
    centers_of_mass = []
    
    for y, x in product(range(N),range(N)):
        if visited[y][x]:
            continue
        if len(grid[y][x]) == 0:
            visited[y][x] = True
            continue
        
        current_component = [(x,y)]
        sum_x = sum_y = 0
        
        components.append(current_component)
        
        i = 0
        while i < len(current_component):
            x, y = current_component[i]

            sum_x += x
            sum_y += y
            
            i+=1
            visited[y][x] = True
            for dx, dy in (0,1),(1,0),(-1,0),(0,-1):
                if not (0 <= y+dy < N and 0 <= x+dx < N):
                    continue
                
                if visited[y+dy][x+dx] or (x+dx, y+dy) in current_component:
                    continue
                
                if len(grid[y+dy][x+dx]) == 0:
                    visited[y+dy][x+dx] = True
                    continue
                
                
                current_component.append((x+dx,y+dy))
            # print('he', *((node[0], node[1],
            #         visited[node[1]][node[0]]) for node in current_component))
            # ...
        # print(current_component)    
        centers_of_mass.append((sum_x / len(current_component), sum_y / len(current_component)))
        
    # print(f'{len(components)=}')
            
    # for line in  visited:
    #     print(*map(int, line))
    
    
    return components, centers_of_mass 
                    
def nearest_grid_point(component: list[tuple[int, int]], point: tuple[int, int]):
    """Using Manhattan distance, find a closet point 
    from the ```component``` list to the ```point```"""
    x, y = point
    
    min_manhattan_distance = float('inf')
    min_x = min_y = -1
    for x0, y0 in component:
        if abs(x - x0) + abs(y - y0) < min_manhattan_distance:
            min_x, min_y = x0, y0
            
            min_manhattan_distance = abs(x - x0) + abs(y - y0)
            
    if min_manhattan_distance < 1:
        return x, y
    
    return min_x, min_y        

def loneliness_score(centers: list[tuple[int, int]]) -> float:
    """using standart deviation of centers of components, we will calcucate the score
    how lonely the components from each other"""
    avg_x = sum(center[0] for center in centers) / len(centers)
    avg_y = sum(center[1] for center in centers) / len(centers)
    
    variance_x = sum((avg_x - center[0])**2 for center in centers) / (len(centers) + 1)
    variance_y = sum((avg_y - center[1])**2 for center in centers) / (len(centers) + 1)
    
    standart_deviation = variance_x**0.5 + variance_y**0.5
    return standart_deviation

def calculate_grade_for_disconnected_components(grid: list[list[list[WordPointer]]]) -> int:
    componets, centers = get_connected_components_and_their_centers(grid)
    # print('centers')
    # print(centers)
    
    nearest_centers = [nearest_grid_point(component, center) for component, center in zip(componets, centers)]
    
    # print('nearest centers')
    # print(nearest_centers)
    
    result = loneliness_score(nearest_centers)
    if 0 < abs(result) < 1:
        result = 1.0
    return (int(result) + len(componets)-1)

#@lru_cache
def fitness(chromosome: Chromosome, logging=False) -> int:
    """main function of the assigment
    
    fitness function will grade particular solution. we asked to find
    first valid solution, so fitness function will grade how 'bad' are 
    the current chromosome solution
    
    bad-ness criteria corresponde with the crossword rules
    
    for each incorrect located letter this function decrease 1 point
    """
    grade = 0
    
    if not logging:
        def print(*args):
            pass
    
    
    grid = generate_grid_with_word_pointers(chromosome)

    # for line in grid:
    #     print(*(len(p) for p in line))
    
    neigbours_grade = 0
    for y in range(N):
        for x in range(N):
            intersects = len(grid[y][x]) 
            
            
            if intersects >= 2:
                grade -= calculate_point_grade_for_non_equal_letters_and_wrong_orintation(grid, x, y)
                
            neigbours_grade += calculate_point_grade_checking_neigbours(grid, x, y)*3
            
            
    print(f'{neigbours_grade=}')
    grade -= neigbours_grade
    # only perpendicular cross should be 
                                
    connected_components_grade = calculate_grade_for_disconnected_components(grid)*10
    print(f'{connected_components_grade=}')
    grade -= connected_components_grade
    return grade                               
                            

def main():
    
    if not os.path.exists('outputs'): 
        os.makedirs('outputs')
    
    i = 1
    while os.path.exists(f'inputs/input{i}.txt'):
        with open(f'outputs/output{i}.txt', 'w') as file:
            with open(f'inputs/input{i}.txt') as inp:
                words = [word.strip() for word in inp.readlines()]
                
                for line in generate_crossword(words):
                    print(*line, file=file)
        
        i+=1


# def main():
#     global N
#     N = 5
    
#     grid = [
#         [[1], [1], [1], [], [], ],
#         [[], [], [], [], [1], ],
#         [[], [], [1], [1], [3], ],
#         [[1], [], [], [], [1], ],
#         [[1], [], [], [], [3], ],
#     ]
    
#     for line in grid:
#         print(*(len(el) for el in line))
    
#     print(f'{calculate_grade_for_disconnected_components(grid)=}')

if __name__ == '__main__':
    main()