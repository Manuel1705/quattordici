#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>
#include <omp.h>

#define N 14


/*
 * Decode a linear index to a tuple of integers
 * the tuple is incremented from left to right
 * -index: index to decode
 * -n_players: number of players, that is the length of the tuple
 * return: pointer to an array of integers representing the state tuple
*/
u_int8_t *decode_state(int index, u_int8_t n_players);

/*
 * Prints a state tuple
 * - state_tuple: pointer to an array of integers representing the state tuple
 * - n_players: number of players, that is the length of the tuple
*/
void print_state_tuple(const u_int8_t *state_tuple, u_int8_t n_players);

/*
 *check if the state is absorbing for the game of quattordici
 * - state: tuple of integers representing the state of each player
 * - n_players: number of players
 * return: true if the state is absorbing, false otherwise
 */
bool is_absorbing_state(const u_int8_t *state_tuple, u_int8_t n_players);


/*
 * Gauss-Jordan matrix inversion
 * - A: pointer to the matrix to invert
 * - n: number of rows and columns of the square matrix A
 * return: pointer to the inverted matrix
*/
double *invert_sqr_matrix(const double *A, int n);


/*
 * Expected absorption time for n_players >= 2
 * - n_players: number of players (1, 2 or 3)
*/
void expected_absorption_time(u_int8_t n_players);

/*
 *
*/
void build_transition_matrix(double *Qn, u_int8_t n_players);

double get_cur_time() {
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return (double) tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main() {
    const int n_players = 3;
    const double start_time = get_cur_time();
    expected_absorption_time(n_players);
    printf("the program took %lf seconds\n", get_cur_time() - start_time);
    return 0;
}

u_int8_t *decode_state(int index, const u_int8_t n_players) {
    u_int8_t *state_tuple = malloc(sizeof(*state_tuple) * n_players);
    if (state_tuple == NULL) exit(EXIT_FAILURE);
    for (int i = 0; i < n_players; i++) {
        state_tuple[i] = index % N;
        index /= N;
    }
    return state_tuple;
}

void print_state_tuple(const u_int8_t *state_tuple, const u_int8_t n_players) {
    if (state_tuple == NULL) exit(EXIT_FAILURE);
    printf("(");
    for (int i = 0; i < n_players; i++) {
        printf("%2d", state_tuple[i]);
        if (i < n_players - 1) printf(", ");
    }
    printf(")");
}

bool is_absorbing_state(const u_int8_t *state_tuple, const u_int8_t n_players) {
    if (state_tuple == NULL) exit(EXIT_FAILURE);
    for (int i = 0; i < n_players; i++) {
        if (state_tuple[i] >= N - 1)
            return true;
    }
    return false;
}

void print_transition_matrix(const double *Q, const int size) {
    if (Q == NULL) exit(EXIT_FAILURE);
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            printf("| %lf ", Q[col + row * size]);
        }
        printf("|\n");
    }
}

void build_transition_matrix(double *Qn, const u_int8_t n_players) {
    if (Qn == NULL) exit(EXIT_FAILURE);
    const int size = (int) pow(N, n_players);


    printf("Building transition matrix of size %d x %d...\n", size, size);
    #pragma omp parallel for num_threads(8) schedule(dynamic) default(none) shared(Qn, size, n_players)
    for (int row = 0; row < size; row++) {
        u_int8_t *state = decode_state(row, n_players);
        if (is_absorbing_state(state, n_players)) {
            free(state);
            continue;
        }
        u_int8_t next_state[n_players];
        for (int player = 0; player < n_players; player++) {
            for (int dice = 1; dice <= 6; dice++) {
                // copy of the current state
                for (int k = 0; k < n_players; k++) {
                    next_state[k] = state[k];
                }

                // updates the next_state for the current player
                // the for() allows to consider all the possibile results of the dice
                // for all the player, one at the time
                next_state[player] += dice;
                // if it's not an absorbing state
                if (next_state[player] < N) {
                    //computes linear index of the new tuple
                    int col = 0;
                    int multiplier = 1;
                    for (int k = 0; k < n_players; k++) {
                        col = col + next_state[k] * multiplier;
                        multiplier *= N;
                    }
                    // updates probability in the transition matrix
                    // 1/6 to go from col state to row state in one turn
                    // every player has 1/n_players probability to throw the dice
                    // so 1/6 * 1/n_players
                    Qn[col + row * size] += 1.0 / (6.0 * n_players);
                }
            }
        }
        free(state);
    }
}

void expected_absorption_time(const u_int8_t n_players) {
    printf("\nCalculating for %d players...\n", n_players);
    const int size = (int) pow(14, n_players);

    if (n_players > 4) {
        printf("Number of players must less the 5.\n");
        return;
    }

    double *Qn = malloc(sizeof(double) * size * size);
    for (int i = 0; i < size * size; i++)
        Qn[i] = 0;

    build_transition_matrix(Qn, n_players);
    if (n_players < 3)
        print_transition_matrix(Qn, size);

    printf("Computing the fundamental matrix...\n");
    double *R = malloc(sizeof(double) * size * size);
    #pragma omp parallel for num_threads(8) schedule(dynamic) default(none) shared(R, Qn, size)
    for (int row = 0; row < size; row++)
        for (int col = 0; col < size; col++)
            R[col + row * size] = (row == col ? 1.0 : 0.0) - Qn[col + row * size];

    double *I = invert_sqr_matrix(R, size);

    printf("Computing the absorption time vector...\n");
    double *hit_time = malloc(sizeof(double) * size);
    #pragma omp parallel for num_threads(8) schedule(dynamic) default(none) shared(hit_time, I, size)
    for (int row = 0; row < size; row++) {
        hit_time[row] = 0.0;
        for (int col = 0; col < size; col++)
            hit_time[row] += I[col + row * size];
    }

    printf("Expected absorption time from each state:\n");
    for (int i = 0; i < size; i++) {
        u_int8_t *state_tuple = decode_state(i, n_players);
        print_state_tuple(state_tuple, n_players);
        printf(": %.6f\n", hit_time[i]);
        free(state_tuple);
    }

    free(hit_time);
    free(I);
    free(R);
    free(Qn);
}


double *invert_sqr_matrix(const double *A, const int n) {
    if (A == NULL) exit(EXIT_FAILURE);
    double *augmented = malloc(sizeof(double) * n * n * 2);
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            // copies A to the left side of augmented
            augmented[row * 2 * n + col] = A[row * n + col];
            // copies I to the right side of augmented
            augmented[row * 2 * n + (col + n)] = (row == col ? 1.0 : 0.0);
        }
    }
    for (int row = 0; row < n; row++) {
        // the pivot is the element at (row, row)
        const double pivot = augmented[row * 2 * n + row];
        // normalization of the pivot row (dividing by the pivot)
        for (int col = 0; col < 2 * n; col++)
            augmented[row * 2 * n + col] /= pivot;

        #pragma omp parallel for num_threads(8) schedule(dynamic) default(none) shared(augmented, n, row)
        for (int row2 = 0; row2 < n; row2++) {
            if (row2 == row) continue;
            const double ratio = augmented[row2 * 2 * n + row];
            for (int col = 0; col < 2 * n; col++)
                augmented[row2 * 2 * n + col] -= ratio * augmented[row * 2 * n + col];
        }
        // waits that every row is updated before moving to the next pivot row
    }

    double *inverse = malloc(sizeof(*inverse) * n * n);
    for (int row = 0; row < n; row++)
        for (int col = 0; col < n; col++)
            inverse[col + row * n] = augmented[col + n + row * 2 * n];
    free(augmented);
    return inverse;
}
