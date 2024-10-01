// GameOfLife.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <SDL.h>
#include <stdio.h>
#include <chrono>
#include <thread>

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
const int CELL_SIZE = 5;
const int GRID_WIDTH = WINDOW_WIDTH / CELL_SIZE;
const int GRID_HEIGHT = WINDOW_HEIGHT / CELL_SIZE;
const int FRAME_DELAY = 100;

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

void initializeGrid(bool* grid) {
    for (int i = 0; i < GRID_WIDTH * GRID_HEIGHT; i++) {
        grid[i] = rand() % 2;
    }
}

int main(int argc, char* argv[]) {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("Game of Life", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    bool* h_grid = new bool[GRID_WIDTH * GRID_HEIGHT];
    bool* h_newGrid = new bool[GRID_WIDTH * GRID_HEIGHT];
    bool* d_grid;
    bool* d_newGrid;

    initializeGrid(h_grid);

    cudaMalloc(&d_grid, GRID_WIDTH * GRID_HEIGHT * sizeof(bool));
    cudaMalloc(&d_newGrid, GRID_WIDTH * GRID_HEIGHT * sizeof(bool));

    cudaMemcpy(d_grid, h_grid, GRID_WIDTH * GRID_HEIGHT * sizeof(bool), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((GRID_WIDTH + blockSize.x - 1) / blockSize.x, (GRID_HEIGHT + blockSize.y - 1) / blockSize.y);

    bool quit = false;
    SDL_Event e;

    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }

        updateGrid << <gridSize, blockSize >> > (d_grid, d_newGrid);
        cudaDeviceSynchronize();

        cudaMemcpy(h_grid, d_newGrid, GRID_WIDTH * GRID_HEIGHT * sizeof(bool), cudaMemcpyDeviceToHost);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);  // Set background to black
        SDL_RenderClear(renderer);

        SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);  // Set color to green
        for (int y = 0; y < GRID_HEIGHT; y++) {
            for (int x = 0; x < GRID_WIDTH; x++) {
                if (h_grid[y * GRID_WIDTH + x]) {
                    SDL_Rect cellRect = { x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE };
                    SDL_RenderFillRect(renderer, &cellRect);
                }
            }
        }

        SDL_RenderPresent(renderer);

        std::swap(d_grid, d_newGrid);
        std::this_thread::sleep_for(std::chrono::milliseconds(FRAME_DELAY));
    }

    cudaFree(d_grid);
    cudaFree(d_newGrid);
    delete[] h_grid;
    delete[] h_newGrid;

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
