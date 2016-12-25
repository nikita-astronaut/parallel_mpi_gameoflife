#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <math.h>

#define ALIVE 'X'
#define DEAD '.'

int GetSqrt(int number) {
    int answer = 1;
    while (answer * answer != number) {
        answer += 1;
    }

    return answer;
}

int CartesianToIndex(int x, int y, int N) {
    if (x < 0) {
        x = x + N;
    } else if (x >= N) {
        x = x - N;
    }
    
    if (y < 0) {
        y = y + N;
    } else if (y >= N) {
        y = y - N;
    }

    return y * N + x;
}

void PrintGrid(char* grid, FILE* f, int N) {
    char * buf = (char *)malloc(N * N * sizeof(char));
    for (int index = 0; index < N; ++index) {
        strncpy(buf, grid + index * N, N);
        buf[N] = 0;
        fprintf(f, "%s\n", buf);
    }
}

int CartesianToProperIndex(int x, int y, int N, int oneDsize) {
    int square_edge = N / oneDsize;
    int x_square_id = x / square_edge;
    int y_square_id = y / square_edge;
    int squares_before = y_square_id * oneDsize + x_square_id;
    
    int x_within_pos = x - square_edge * x_square_id;
    int y_within_pos = y - square_edge * y_square_id;

    return squares_before * square_edge * square_edge + y_within_pos * square_edge + x_within_pos;
}

// as we're ging to perform 2D decomposition, we need to store elements properly for using mpi_scatter
// first N * N / size elements belong to upper-left square, next N * N / size elements belong to upper-next-to-left square and so on
char * GetInitialState(int N, char * filename, int size) {    
    char * grid_raw = (char *)malloc(N * N * sizeof(char));
    FILE* input_file = fopen(filename, "r");
    for (int i = 0; i < N; ++i) {
        fscanf(input_file, "%s", grid_raw + i * N);
    }
    fclose(input_file);

    char * grid_proper = (char *)malloc(N * N * sizeof(char));
    int oneDsize = GetSqrt(size);

    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            *(grid_proper + CartesianToProperIndex(x, y, N, oneDsize)) = *(grid_raw + CartesianToIndex(x, y, N));
        }
    }

    return grid_proper;
}

void FillBulk(char * * local_grid, char * local_grid_preliminary, int square_edge) {
    for (int x = 0; x < square_edge; ++x) {
        for (int y = 0; y < square_edge; ++y) {
            // this is merely a shift x -> x + 1, y -> y + 1, essentially to create a free edge that will be convenient
            *(*local_grid + (square_edge + 2) * (y + 1) + (x + 1)) = *(local_grid_preliminary + square_edge * y + x);
        }
    }
}

// CHAR
void AllocateMemoryCHAR(int N, int oneDsize, 
    char * * up_send_buffer, char * * down_send_buffer, 
    char * * left_send_buffer, char * * right_send_buffer, 
    char * * up_left_send_buffer, char * * up_right_send_buffer, 
    char * * down_left_send_buffer, char * * down_right_send_buffer) {
    int square_edge = N / oneDsize;

    *up_send_buffer = (char *)malloc((square_edge + 2) * sizeof(char));
    *down_send_buffer = (char *)malloc((square_edge + 2) * sizeof(char));
    *left_send_buffer = (char *)malloc((square_edge + 2) * sizeof(char));
    *right_send_buffer = (char *)malloc((square_edge + 2) * sizeof(char));
    for (int i = 0; i < square_edge + 2; ++i) {
        *(*up_send_buffer + i) = DEAD;
        *(*down_send_buffer + i) = DEAD;
        *(*left_send_buffer + i) = DEAD;
        *(*right_send_buffer + i) = DEAD;
    }

    *up_left_send_buffer = (char *)malloc(sizeof(char));
    *(*up_left_send_buffer) = DEAD;
    *up_right_send_buffer = (char *)malloc(sizeof(char));
    *(*up_right_send_buffer) = DEAD;
    *down_left_send_buffer = (char *)malloc(sizeof(char));
    *(*down_left_send_buffer) = DEAD;
    *down_right_send_buffer = (char *)malloc(sizeof(char));
    *(*down_right_send_buffer) = DEAD;
}

// CHAR
void FillBoundariesToSendCHAR(char * data, int rank, int N, int oneDsize, 
    char * * up_send_buffer, char * * down_send_buffer, 
    char * * left_send_buffer, char * * right_send_buffer, 
    char * * up_left_send_buffer, char * * up_right_send_buffer, 
    char * * down_left_send_buffer, char * * down_right_send_buffer) {
    int square_edge = N / oneDsize;

    // for example, up_send_buffer is square_edge + 2 long = edge + 2 corners. only edge information is useful here
    for (int x = 1; x < square_edge + 1; ++x) {
        *(*up_send_buffer + x) = *(data + 1 * (square_edge + 2) + x);
    }

    for (int x = 1; x < square_edge + 1; ++x) {
        *(*down_send_buffer + x) = *(data + square_edge * (square_edge + 2) + x);
    }

    for (int y = 1; y < square_edge + 1; ++y) {
        *(*left_send_buffer + y) = *(data + y * (square_edge + 2) + 1);
    }

    for (int y = 1; y < square_edge + 1; ++y) {
        *(*right_send_buffer + y) = *(data + y * (square_edge + 2) + square_edge);
    }

    *(*up_left_send_buffer) = *(data + (square_edge + 2) + 1);

    *(*up_right_send_buffer) = *(data + (square_edge + 2) + square_edge);

    *(*down_left_send_buffer) = *(data + square_edge * (square_edge + 2) + 1);

    *(*down_right_send_buffer) = *(data + square_edge * (square_edge + 2) + square_edge);
}

int * InitializeChangedStatus(int square_edge) {
    int * changed_status = (int *)malloc((square_edge + 2) * (square_edge + 2) * sizeof(int));
    for (int x = 0; x < square_edge + 2; ++x) {
        for (int y = 0; y < square_edge + 2; ++y) {
            *(changed_status + (square_edge + 2) * y + x) = -1;
        }
    }

    return changed_status;
}

int GetAliveCount(int x, int y, int square_edge, char * grid_state) {
    int counter = 0;
    for (int d_x = -1; d_x <= 1; ++d_x) {
        for (int d_y = -1; d_y <= 1; ++d_y) {
            if (*(grid_state + (y + d_y) * (square_edge + 2) + (x + d_x)) == ALIVE) {
                if (d_x != 0 || d_y != 0) {
                    ++counter;
                }
            }
        }
    }

    return counter;
}

void MarkChanged(int x, int y, int square_edge, int * * changed_state_buffer, int iterration) {
    for (int d_x = -1; d_x <= 1; ++d_x) {
        for (int d_y = -1; d_y <= 1; ++d_y) {
            *(*changed_state_buffer + (square_edge + 2) * (y + d_y) + (x + d_x)) = iterration;
        }
    }
}

void ProcessPoint(int x, int y, int square_edge, char * grid_state, 
    char * * grid_state_buffer, int * changed_status, 
    int * * changed_state_buffer, int iterration, int anyway) {

    if (*(changed_status + y * (square_edge + 2) + x) == iterration - 1 || anyway == 1) {
        int alive_count = GetAliveCount(x, y, square_edge, grid_state);

        if (alive_count == 3 || (alive_count == 2 && *(grid_state + y * (square_edge + 2) + x) == ALIVE)) {
            *(*grid_state_buffer + (square_edge + 2) * y + x) = ALIVE;
        } else {
            *(*grid_state_buffer + (square_edge + 2) * y + x) = DEAD;
        }

        if (*(*grid_state_buffer + (square_edge + 2) * y + x) != *(grid_state + (square_edge + 2) * y + x)) {
            MarkChanged(x, y, square_edge, changed_state_buffer, iterration);
        }
    }
}

void AddReceivedDataCHAR(int square_edge, char * * grid_state, char * up_recv_buffer, char * down_recv_buffer, 
            char * left_recv_buffer, char * right_recv_buffer, char * up_left_recv_buffer, 
            char * up_right_recv_buffer, char * down_left_recv_buffer, char * down_right_recv_buffer) {
    int x, y;
    
    x = 0;
    y = 0;
    *(*grid_state + y * (square_edge + 2) + x) = *(up_left_recv_buffer);

    y = 0;
    for (x = 1; x < square_edge + 1; ++x) {
        *(*grid_state + y * (square_edge + 2) + x) = *(up_recv_buffer + x);
    }

    x = square_edge + 1;
    y = 0;
    *(*grid_state + y * (square_edge + 2) + x) = *(up_right_recv_buffer);

    x = square_edge + 1;
    for (y = 1; y < square_edge + 1; ++y) {
        *(*grid_state + y * (square_edge + 2) + x) = *(right_recv_buffer + y);
    }

    x = square_edge + 1;
    y = square_edge + 1;
    *(*grid_state + y * (square_edge + 2) + x) = *(down_right_recv_buffer);

    y = square_edge + 1;
    for (x = 1; x < square_edge + 1; ++x) {
        *(*grid_state + y * (square_edge + 2) + x) = *(down_recv_buffer + x);
    }

    x = 0;
    y = square_edge + 1;
    *(*grid_state + y * (square_edge + 2) + x) = *(down_left_recv_buffer);

    x = 0;
    for (y = 1; y < square_edge + 1; ++y) {
        *(*grid_state + y * (square_edge + 2) + x) = *(left_recv_buffer + y);
    }
}

char * GetFinalField(char * raw_final_field, int square_edge, int oneDsize) {
    char * final_field = (char *)malloc(square_edge * square_edge * oneDsize * oneDsize * sizeof(char));
    int total_size = (square_edge + 2) * (square_edge + 2) * oneDsize * oneDsize;
    int box_size = (square_edge + 2) * (square_edge + 2);

    for (int index = 0; index < total_size; ++index) {
        int box_id = index / box_size;
        int box_x = box_id % oneDsize;
        int box_y = box_id / oneDsize;

        int coord_index = index - box_id * box_size;
        int x = coord_index % (square_edge + 2);
        int y = coord_index / (square_edge + 2);

        if (x > 0 && x < square_edge + 1 && y > 0 && y < square_edge + 1) {
            --x;
            --y;
            *(final_field + (box_y * square_edge + y) * square_edge * oneDsize + box_x * square_edge + x) = *(raw_final_field + index);
        }
    }

    return final_field;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s N input_file iterations output_file\n", argv[0]);
        return 1;
    }

    int argc_mpi;
    char **argv_mpi;

    int N = atoi(argv[1]);
    int iterrations = atoi(argv[3]);
    char * grid;
    int rank;
    int size;
    int tag = 1;

    MPI_Init(&argc, &argv);

    MPI_Request reqs[16];
    MPI_Status stats[16];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // read initial state in form appropriate for 2D scatter
    if (rank == 0) {
        grid = GetInitialState(N, argv[2], size);
    }

    int oneDsize = GetSqrt(size);
    int square_edge = N / oneDsize;
    char * grid_state_preliminary = (char *)malloc((N * N / size) * sizeof(char));

    MPI_Scatter(grid, N * N / size, MPI_CHAR, grid_state_preliminary, N * N / size, MPI_CHAR, 0, MPI_COMM_WORLD);

    char * grid_state = (char *)malloc((N / oneDsize + 2) * (N / oneDsize + 2) * sizeof(char));
    FillBulk(&grid_state, grid_state_preliminary, N / oneDsize);

    int * changed_state = InitializeChangedStatus(N / oneDsize);

    // create and allocate buffers to send grid_state
    char * up_send_buffer;
    char * down_send_buffer; 
    char * left_send_buffer; 
    char * right_send_buffer; 
    char * up_left_send_buffer;
    char * up_right_send_buffer;
    char * down_left_send_buffer;
    char * down_right_send_buffer;
    AllocateMemoryCHAR(N, oneDsize, &up_send_buffer, &down_send_buffer, 
            &left_send_buffer, &right_send_buffer, &up_left_send_buffer, &up_right_send_buffer,
            &down_left_send_buffer, &down_right_send_buffer);   

    // create and allocate buffers to receive grid_state
    char * up_recv_buffer;
    char * down_recv_buffer;
    char * left_recv_buffer;
    char * right_recv_buffer; 
    char * up_left_recv_buffer;
    char * up_right_recv_buffer;
    char * down_left_recv_buffer;
    char * down_right_recv_buffer;
    AllocateMemoryCHAR(N, oneDsize, &up_recv_buffer, &down_recv_buffer, 
            &left_recv_buffer, &right_recv_buffer, &up_left_recv_buffer, &up_right_recv_buffer,
            &down_left_recv_buffer, &down_right_recv_buffer);

    // ----------------- CREATE BUFFER COPIES OF BOTH ARRAYS ---------------//
    char * grid_state_buffer = (char *)malloc((N / oneDsize + 2) * (N / oneDsize + 2) * sizeof(char));
    int * changed_state_buffer = (int *)malloc((N / oneDsize + 2) * (N / oneDsize + 2) * sizeof(int));
    /*
    for (int x = 0; x < square_edge + 2; ++x) {
        for (int y = 0; y < square_edge + 2; ++y) {
            *(grid_state_buffer + (square_edge + 2) * y + x) = *(grid_state + (square_edge + 2) * y + x);
            *(changed_state_buffer + (square_edge + 2) * y + x) = *(changed_state + (square_edge + 2) * y + x);
        }
    }
    */
    // -------------------- START OF ITERRATIONS ---------------------//
    for (int iterration = 0; iterration < iterrations; ++iterration) {

        FillBoundariesToSendCHAR(grid_state, rank, N, oneDsize, &up_send_buffer, &down_send_buffer, 
            &left_send_buffer, &right_send_buffer, &up_left_send_buffer, &up_right_send_buffer,
            &down_left_send_buffer, &down_right_send_buffer);

        int x_box_coord = rank % oneDsize;
        int y_box_coord = rank / oneDsize;
        int square_edge = N / oneDsize;

        // --------------------- CAST OF ALL SEND CALLS ------------------------ //
        MPI_Isend(left_send_buffer, square_edge + 2, MPI_CHAR, 
            CartesianToIndex(x_box_coord - 1, y_box_coord, oneDsize), 0, MPI_COMM_WORLD, &reqs[0]);

        MPI_Isend(right_send_buffer, square_edge + 2, MPI_CHAR, 
            CartesianToIndex(x_box_coord + 1, y_box_coord, oneDsize), 1, MPI_COMM_WORLD, &reqs[1]);

        MPI_Isend(up_send_buffer, square_edge + 2, MPI_CHAR, 
            CartesianToIndex(x_box_coord, y_box_coord - 1, oneDsize), 2, MPI_COMM_WORLD, &reqs[2]);

        MPI_Isend(down_send_buffer, square_edge + 2, MPI_CHAR, 
            CartesianToIndex(x_box_coord, y_box_coord + 1, oneDsize), 3, MPI_COMM_WORLD, &reqs[3]);

        MPI_Isend(up_left_send_buffer, 1, MPI_CHAR, 
            CartesianToIndex(x_box_coord - 1, y_box_coord - 1, oneDsize), 4, MPI_COMM_WORLD, &reqs[4]);

        MPI_Isend(up_right_send_buffer, 1, MPI_CHAR, 
            CartesianToIndex(x_box_coord + 1, y_box_coord - 1, oneDsize), 5, MPI_COMM_WORLD, &reqs[5]);

        MPI_Isend(down_left_send_buffer, 1, MPI_CHAR, 
            CartesianToIndex(x_box_coord - 1, y_box_coord + 1, oneDsize), 6, MPI_COMM_WORLD, &reqs[6]);

        MPI_Isend(down_right_send_buffer, 1, MPI_CHAR, 
            CartesianToIndex(x_box_coord + 1, y_box_coord + 1, oneDsize), 7, MPI_COMM_WORLD, &reqs[7]);
        // --------------------- END OF CAST OF ALL SEND CALLS ------------------------ //

        // --------------------- CAST OF ALL RECEIVE CALLS ------------------------- //
        MPI_Irecv(left_recv_buffer, square_edge + 2, MPI_CHAR, 
            CartesianToIndex(x_box_coord - 1, y_box_coord, oneDsize), 1, MPI_COMM_WORLD, &reqs[8]);

        MPI_Irecv(right_recv_buffer, square_edge + 2, MPI_CHAR, 
            CartesianToIndex(x_box_coord + 1, y_box_coord, oneDsize), 0, MPI_COMM_WORLD, &reqs[9]);

        MPI_Irecv(up_recv_buffer, square_edge + 2, MPI_CHAR, 
            CartesianToIndex(x_box_coord, y_box_coord - 1, oneDsize), 3, MPI_COMM_WORLD, &reqs[10]);

        MPI_Irecv(down_recv_buffer, square_edge + 2, MPI_CHAR, 
            CartesianToIndex(x_box_coord, y_box_coord + 1, oneDsize), 2, MPI_COMM_WORLD, &reqs[11]);

        MPI_Irecv(up_left_recv_buffer, 1, MPI_CHAR, 
            CartesianToIndex(x_box_coord - 1, y_box_coord - 1, oneDsize), 7, MPI_COMM_WORLD, &reqs[12]);

        MPI_Irecv(up_right_recv_buffer, 1, MPI_CHAR, 
            CartesianToIndex(x_box_coord + 1, y_box_coord - 1, oneDsize), 6, MPI_COMM_WORLD, &reqs[13]);

        MPI_Irecv(down_left_recv_buffer, 1, MPI_CHAR, 
            CartesianToIndex(x_box_coord - 1, y_box_coord + 1, oneDsize), 5, MPI_COMM_WORLD, &reqs[14]);

        MPI_Irecv(down_right_recv_buffer, 1, MPI_CHAR, 
            CartesianToIndex(x_box_coord + 1, y_box_coord + 1, oneDsize), 4, MPI_COMM_WORLD, &reqs[15]);
        
        // ----------------------- END OF CAST OF ALL RECEIVE CALLS ------------------------ //

        // ----------------- DO JOB WITHIN BULK --------------- //
        for (int x = 2; x < square_edge; ++x) {
            for (int y = 2; y < square_edge; ++y) {
                ProcessPoint(x, y, square_edge, grid_state, &grid_state_buffer, 
                    changed_state, &changed_state_buffer, iterration, 0);
            }
        }

        // ----------------- W8T FOR ALL THE TRANSITIONS TO OCCUR ------------------ //
        MPI_Waitall(16, reqs, stats);
        
        // ----------------- COPY RECEIVED DATA TO CURRENT STATE ------------------- //
        AddReceivedDataCHAR(square_edge, &grid_state, up_recv_buffer, down_recv_buffer, 
            left_recv_buffer, right_recv_buffer, up_left_recv_buffer, 
            up_right_recv_buffer, down_left_recv_buffer, down_right_recv_buffer);

        // ------------------ FINISH ON-EDGE CALCULATIONS ------------------------- //
        int x, y;

        y = 1;
        for (x = 1; x < square_edge; ++x) {
            ProcessPoint(x, y, square_edge, grid_state, &grid_state_buffer, 
                    changed_state, &changed_state_buffer, iterration, 1);
        }

        x = square_edge;
        for (y = 1; y < square_edge; ++y) {
            ProcessPoint(x, y, square_edge, grid_state, &grid_state_buffer, 
                    changed_state, &changed_state_buffer, iterration, 1);
        }

        y = square_edge;
        for (x = square_edge; x > 1; -- x) {
            ProcessPoint(x, y, square_edge, grid_state, &grid_state_buffer, 
                    changed_state, &changed_state_buffer, iterration, 1);
        }

        x = 1;
        for (y = square_edge; y > 1; --y) {
            ProcessPoint(x, y, square_edge, grid_state, &grid_state_buffer, 
                    changed_state, &changed_state_buffer, iterration, 1);
        }
        
        // ------------------- SWAP CURRENT DATA AND BUTTERS TO PROCEED ----------------- //
        char * char_swapper;
        char_swapper = grid_state;
        grid_state = grid_state_buffer;
        grid_state_buffer = char_swapper;

        int * int_swapper;
        int_swapper = changed_state;
        changed_state = changed_state_buffer;
        changed_state_buffer = int_swapper;
    }

    // ------------------- COLLECT DATA FROM DISTINCT PROCESSES ------------------- //
    char * raw_final_field;
    if (rank == 0) {
        raw_final_field = (char *)malloc((square_edge + 2) * (square_edge + 2) * size * sizeof(char));
    }

    MPI_Gather(grid_state, (square_edge + 2) * (square_edge + 2), MPI_CHAR,
        raw_final_field, (square_edge + 2) * (square_edge + 2), MPI_CHAR,
        0, MPI_COMM_WORLD);
    if (rank == 0) {
        char * final_field = GetFinalField(raw_final_field, square_edge, oneDsize);

        FILE* output = fopen(argv[4], "w");
        PrintGrid(final_field, output, N);
    }

    MPI_Finalize();

    free(grid_state);
    free(grid_state_buffer);
    free(changed_state);
    free(changed_state_buffer);

    return 0;
}
