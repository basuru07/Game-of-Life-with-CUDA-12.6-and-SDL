#include <stdio.h>
#include <cuda_runtime.h>
#include <SDL.h>

// Define constants for the grid size and cell size
#define WIDTH 800
#define HEIGHT 800
#define CELL_SIZE 30  // Increased for larger cell size
#define GRID_WIDTH (WIDTH / CELL_SIZE)
#define GRID_HEIGHT (HEIGHT / CELL_SIZE)
#define TILE_SIZE 16  // Size of the tile for CUDA blocks

// Macro to check for CUDA errors
#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernel to update the cells in the Game of Life
__global__ void updateCells(int* currentGrid, int* nextGrid) {
    // Shared memory for tiles
    __shared__ int sharedGrid[TILE_SIZE + 2][TILE_SIZE + 2];

    // Calculate global and local thread coordinates
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int tx = threadIdx.x + 1;  // Local x index (1-based for ghost cells)
    int ty = threadIdx.y + 1;  // Local y index (1-based for ghost cells)

    // Return if the thread is out of bounds
    if (x >= GRID_WIDTH || y >= GRID_HEIGHT) return;

    // Load the current cell into shared memory
    sharedGrid[ty][tx] = currentGrid[y * GRID_WIDTH + x];

    // Handle boundary conditions by loading ghost cells
    if (threadIdx.x == 0) {
        // Left ghost cell
        sharedGrid[ty][0] = currentGrid[y * GRID_WIDTH + (x - 1 + GRID_WIDTH) % GRID_WIDTH];
    }
    if (threadIdx.x == TILE_SIZE - 1) {
        // Right ghost cell
        sharedGrid[ty][TILE_SIZE + 1] = currentGrid[y * GRID_WIDTH + (x + 1) % GRID_WIDTH];
    }
    if (threadIdx.y == 0) {
        // Top ghost cell
        sharedGrid[0][tx] = currentGrid[((y - 1 + GRID_HEIGHT) % GRID_HEIGHT) * GRID_WIDTH + x];
    }
    if (threadIdx.y == TILE_SIZE - 1) {
        // Bottom ghost cell
        sharedGrid[TILE_SIZE + 1][tx] = currentGrid[((y + 1) % GRID_HEIGHT) * GRID_WIDTH + x];
    }

    // Synchronize threads to ensure all shared memory is loaded
    __syncthreads();

    // Count the number of neighbors
    int neighbors = 0;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            if (dx == 0 && dy == 0) continue;  // Skip the cell itself
            neighbors += sharedGrid[ty + dy][tx + dx];  // Sum neighbors
        }
    }

    // Determine the new state of the cell based on the rules of Conway's Game of Life
    int cellState = sharedGrid[ty][tx];
    if (cellState == 1 && (neighbors < 2 || neighbors > 3)) {
        nextGrid[y * GRID_WIDTH + x] = 0;  // Cell dies
    }
    else if (cellState == 0 && neighbors == 3) {
        nextGrid[y * GRID_WIDTH + x] = 1;  // Cell becomes alive
    }
    else {
        nextGrid[y * GRID_WIDTH + x] = cellState;  // Stays the same
    }
}

// Function to initialize the grid with random values
void initializeGrid(int* grid) {
    for (int y = 0; y < GRID_HEIGHT; ++y) {
        for (int x = 0; x < GRID_WIDTH; ++x) {
            grid[y * GRID_WIDTH + x] = rand() % 2;  // Randomly set cell state (alive or dead)
        }
    }
}

// Function to render the grid using SDL
void renderGrid(SDL_Renderer* renderer, int* grid) {
    // Clear the screen with black color
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    // Set the draw color for alive cells
    SDL_SetRenderDrawColor(renderer, 128, 0, 128, 255);

    // Draw each cell based on its state
    for (int y = 0; y < GRID_HEIGHT; ++y) {
        for (int x = 0; x < GRID_WIDTH; ++x) {
            if (grid[y * GRID_WIDTH + x] == 1) {
                SDL_Rect cellRect = { x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE };
                SDL_RenderFillRect(renderer, &cellRect);  // Fill the cell rectangle
            }
        }
    }

    // Present the rendered frame
    SDL_RenderPresent(renderer);
}

int main(int argc, char** argv) {
    // Initialize SDL
    SDL_Init(SDL_INIT_VIDEO);

    // Create SDL window and renderer
    SDL_Window* window = SDL_CreateWindow("Conway's Game of Life", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    // Allocate memory for the grid on the host
    int* hostGrid, * hostNextGrid;
    hostGrid = (int*)malloc(GRID_WIDTH * GRID_HEIGHT * sizeof(int));
    hostNextGrid = (int*)malloc(GRID_WIDTH * GRID_HEIGHT * sizeof(int));

    // Initialize the grid with random values
    initializeGrid(hostGrid);

    // Allocate memory for the grid on the device
    int* devGrid, * devNextGrid;
    cudaMalloc((void**)&devGrid, GRID_WIDTH * GRID_HEIGHT * sizeof(int));
    cudaMalloc((void**)&devNextGrid, GRID_WIDTH * GRID_HEIGHT * sizeof(int));

    // Set up the block and grid sizes for the CUDA kernel
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((GRID_WIDTH + TILE_SIZE - 1) / TILE_SIZE, (GRID_HEIGHT + TILE_SIZE - 1) / TILE_SIZE);

    bool running = true;  // Flag to control the main loop
    SDL_Event event;  // SDL event structure

    // CUDA events for timing the kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Main loop
    while (running) {
        // Handle SDL events
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;  // Exit on quit event
            }
        }

        // Copy the current grid from host to device
        cudaMemcpy(devGrid, hostGrid, GRID_WIDTH * GRID_HEIGHT * sizeof(int), cudaMemcpyHostToDevice);

        // Start measuring time
        cudaEventRecord(start, 0);

        // Launch the CUDA kernel to update cells
        updateCells << <gridSize, blockSize >> > (devGrid, devNextGrid);
        cudaCheckError();  // Check for any CUDA errors

        // Stop measuring time
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Calculate the elapsed time for kernel execution
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Kernel execution time: %f ms\n", milliseconds);

        // Copy the next grid back from device to host
        cudaMemcpy(hostNextGrid, devNextGrid, GRID_WIDTH * GRID_HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

        // Swap the grids for the next iteration
        int* temp = hostGrid;
        hostGrid = hostNextGrid;
        hostNextGrid = temp;

        // Render the grid
        renderGrid(renderer, hostGrid);

        // Delay for a short time to control the frame rate
        SDL_Delay(100);
    }

    // Free device memory
    cudaFree(devGrid);
    cudaFree(devNextGrid);

    // Free host memory
    free(hostGrid);
    free(hostNextGrid);

    // Clean up SDL resources
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;  // Exit the program
}
