# Game of Life with CUDA 12.6 and SDL

This project implements Conway's Game of Life using CUDA 12.6 for parallel processing and SDL for graphical rendering. The Game of Life is a cellular automaton where cells live, die, or multiply based on certain rules.

## Prerequisites

To run this project, you'll need:

- **CUDA 12.6**: Ensure that you have CUDA 12.6 installed on your system. You can download it from the [NVIDIA website](https://developer.nvidia.com/cuda-toolkit).
- **Visual Studio 2022**: This project was built using Visual Studio 2022. You can download it from [Microsoft's website](https://visualstudio.microsoft.com/).
- **SDL2**: Install the Simple DirectMedia Layer (SDL) for graphical rendering. You can get it from [SDL's official website](https://www.libsdl.org/).

## Setup

1. Install **CUDA 12.6** on your system and configure the environment paths.
2. Open **Visual Studio 2022** and create a new project:
   - Go to **File > New > Project**.
   - Select **CUDA 12.6 Runtime** from the available templates.
3. Add **SDL2** to your project:
   - Download the SDL2 development libraries.
   - Include the SDL headers in your project and link the SDL2 libraries.
4. Clone this repository and open the project in Visual Studio.

## Code Overview

This project uses CUDA to update the state of the Game of Life grid and SDL to render the grid on a window.

- The grid is a 2D array where each cell is either alive (`1`) or dead (`0`).
- CUDA is used to calculate the next generation of the grid in parallel, which speeds up the computation for large grids.
- SDL is used to draw the grid on the screen, with each live cell represented as a green square.

### Key Files

- **GameOfLife.cu**: Contains the main logic for the Game of Life, including CUDA kernels for updating the grid and SDL code for rendering.

### CUDA Kernel

The core of the simulation is the CUDA kernel `updateGrid`, which computes the next state of each cell based on its neighbors:

```cpp
__global__ void updateGrid(bool* d_grid, bool* d_newGrid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= GRID_WIDTH || y >= GRID_HEIGHT) return;

    int aliveNeighbors = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = (x + dx + GRID_WIDTH) % GRID_WIDTH;
            int ny = (y + dy + GRID_HEIGHT) % GRID_HEIGHT;
            aliveNeighbors += d_grid[ny * GRID_WIDTH + nx];
        }
    }

    bool currentCell = d_grid[y * GRID_WIDTH + x];
    bool newState = (currentCell && (aliveNeighbors == 2 || aliveNeighbors == 3)) ||
        (!currentCell && aliveNeighbors == 3);
    d_newGrid[y * GRID_WIDTH + x] = newState;
}
```
## How to Run

1. Build the project in **Visual Studio 2022**.
2. Make sure that you have **CUDA** and **SDL** properly configured in your environment.
3. Run the project, and a window will appear displaying the Game of Life simulation.
   - Green squares represent live cells.
   - The grid updates every 100 milliseconds.



