def display_Path_to_Princess(n, grid):
    # Find bot ('m') and princess ('p') positions
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'm':
                bot = (i, j)
            elif grid[i][j] == 'p':
                princess = (i, j)
    
    # Calculate moves
    row_diff = princess[0] - bot[0]
    col_diff = princess[1] - bot[1]
    
    # Move vertically
    if row_diff > 0:
        for _ in range(row_diff):
            print("DOWN")
    elif row_diff < 0:
        for _ in range(-row_diff):
            print("UP")
    
    # Move horizontally
    if col_diff > 0:
        for _ in range(col_diff):
            print("RIGHT")
    elif col_diff < 0:
        for _ in range(-col_diff):
            print("LEFT")


# Sample Input
n = 3
grid = [
    "---",
    "-m-",
    "p--"
]

display_Path_to_Princess(n, grid)
