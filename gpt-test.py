def get_connected_components_and_their_centers(grid):
    global N
    visited = [[False] * N for _ in range(N)]
    
    components = []
    centers_of_mass = []
    for y in range(N):
        for x in range(N):
            print(f"Checking position: x = {x}, y = {y}")  # Debug: Check each position
            if visited[y][x]:
                print(f"Already visited: x = {x}, y = {y}")  # Debug: Skip visited positions
                continue
            if len(grid[y][x]) == 0:
                visited[y][x] = True
                continue
            
            current_component = [(x, y)]
            sum_x = sum_y = 0
            
            i = 0
            while i < len(current_component):
                x, y = current_component[i]
                print(f"Processing node: x = {x}, y = {y}")  # Debug: Processing node

                sum_x += x
                sum_y += y
                
                i += 1
                visited[y][x] = True
                for dx, dy in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                    if not (0 <= y + dy < N and 0 <= x + dx < N):
                        continue
                    if visited[y + dy][x + dx] or (x + dx, y + dy) in current_component:
                        continue
                    if len(grid[y + dy][x + dx]) == 0:
                        visited[y + dy][x + dx] = True
                        continue
                    print(f"Adding to component: x = {x + dx}, y = {y + dy}")  # Debug: Adding new node to component
                    current_component.append((x + dx, y + dy))
            
            print(f"Current component: {current_component}")  # Debug: Completed component
            components.append(current_component)
            centers_of_mass.append((sum_x / len(current_component), sum_y / len(current_component)))
        
    print(f'Number of components: {len(components)}')  # Debug: Number of components
    for comp in components:
        print(f"Component: {comp}")  # Debug: List components
    return components, centers_of_mass 

def main():
    global N
    N = 5
    
    grid = [
        [[], [1], [1], [], [1], ],
        [[1], [], [1], [], [], ],
        [[1], [], [2], [], [3], ],
        [[1], [], [2], [], [3], ],
        [[1], [], [2], [], [3], ],
    ]
    
    for line in grid:
        print(*(len(el) for el in line))
    
    get_connected_components_and_their_centers(grid)

# Running your main function
main()
