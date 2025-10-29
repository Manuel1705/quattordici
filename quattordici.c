#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>
#include <stdint.h>
#include <omp.h>
#include <string.h>

#define N 14

/**
 * Decode a linear index to a tuple of integers
 * the tuple is incremented from left to right
 * @param index index to decode
 * @param n_players number of players, that is the length of the tuple
 * @returns pointer to an array of integers representing the state tuple
*/
u_int8_t *decode_state(int index, u_int8_t n_players);

/**
 * Prints a state tuple
 * @param state_tuple pointer to an array of integers representing the state tuple
 * @param n_players number of players, that is the length of the tuple
*/
void print_state_tuple(const u_int8_t *state_tuple, u_int8_t n_players);

/**
 *check if the state is absorbing for the game of quattordici
 * @param state_tuple of integers representing the state of each player
 * @param n_players number of players
 * @returns true if the state is absorbing, false otherwise
 */
bool is_absorbing_state(const u_int8_t *state_tuple, u_int8_t n_players);


/**
 * Gauss-Jordan matrix inversion
 * @param A pointer to the matrix to invert
 * @param n number of rows and columns of the square matrix A
 * @returns pointer to the inverted matrix
*/
double *invert_sqr_matrix(const double *A, int n);


/**
 * Expected absorption time for n_players >= 2
 * @param n_players number of players (1, 2 or 3)
*/
void expected_absorption_time(u_int8_t n_players);

void build_transition_matrix(double *Qn, u_int8_t n_players);

/**
 * Calculates the mean of an array of values.
 * @param values array of values
 * @param n  length of the array
 * @return the mean of the array
 */
double mean(const u_int8_t *values, long n);

/**
 * Calculates the population standard deviation of an array of values.
 * @param values array of values
 * @param n number of elements in the array
 * @param avg mean of the array
 * @return - Standard deviation of the array
 */
double p_st_dev(const uint8_t *values, long n, double avg);

/**
 * Montecarlo simulation of the game of quattordici
 * simulates the game for increasing number of players
 * and calculates the average number of turns to finish the game
 * @param attempts number of attempts to simulate
 * @param max_players maximum number of players to simulate
 */
void montecarlo_simulation(long int attempts, u_int8_t max_players);

double get_cur_time() {
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return (double) tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main() {
    const int n_players = 2;
    const double start_time = get_cur_time();
    expected_absorption_time(n_players);
    printf("the program took %lf seconds\n", get_cur_time() - start_time);
    montecarlo_simulation(10000000, 10);
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
        if (state_tuple[i] >= N)
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
    const int total_states = (int) pow(N, n_players);

    printf("Building transition matrix (only transient states) for %d players...\n", n_players);
    #pragma omp parallel for num_threads(8) schedule(dynamic) default(none) shared(Qn, total_states, n_players)
    for (int row = 0; row < total_states; row++) {
        u_int8_t *state = decode_state(row, n_players);
        if (is_absorbing_state(state, n_players)) {
            free(state);
            continue; // skip absorbing states
        }

        u_int8_t next_state[n_players];

        // for every player
        for (int player = 0; player < n_players; player++) {
            // for every possible dice result
            for (int dice = 1; dice <= 6; dice++) {
                // copy of the current state
                memcpy(next_state, state, n_players * sizeof(u_int8_t));
                // updates the next_state
                next_state[player] += dice;

                // if it's not an absorbing state
                if (next_state[player] < N) {
                    //computes linear index of the new tuple
                    int index = 0;
                    int multiplier = 1;
                    for (int k = 0; k < n_players; k++) {
                        index += next_state[k] * multiplier;
                        multiplier *= N;
                    }
                    // updates probability in the transition matrix
                    // 1/6 to go from col state to row state in one turn
                    // every player has 1/n_players probability to throw the dice
                    // so 1/6 * 1/n_players
                    #pragma omp atomic
                    Qn[index + row * total_states] += 1.0 / (6.0 * n_players);
                } else {
                    break; // no need to check higher dice values for this player
                }
            }
        }
        free(state);
    }
}

void expected_absorption_time(const u_int8_t n_players) {
    printf("\nCalculating expected absorption time (Markov) for %d players...\n", n_players);
    const int total_states = (int) pow(N, n_players);

    double *Qn = calloc(total_states * total_states, sizeof(double));
    if (Qn == NULL) exit(EXIT_FAILURE);

    build_transition_matrix(Qn, n_players);
    if (n_players == 1)
        print_transition_matrix(Qn, total_states);

    printf("Computing the fundamental matrix...\n");
    double *R = malloc(sizeof(double) * total_states * total_states);
    #pragma omp parallel for num_threads(8) schedule(dynamic) default(none) shared(R, Qn, total_states)
    for (int row = 0; row < total_states; row++)
        for (int col = 0; col < total_states; col++)
            R[col + row * total_states] = (row == col ? 1.0 : 0.0) - Qn[col + row * total_states];

    double *I = invert_sqr_matrix(R, total_states);

    printf("Computing the absorption time vector...\n");
    double *hit_time = malloc(sizeof(double) * total_states);
    #pragma omp parallel for num_threads(8) schedule(dynamic) default(none) shared(hit_time, I, total_states)
    for (int row = 0; row < total_states; row++) {
        hit_time[row] = 0.0;
        for (int col = 0; col < total_states; col++)
            hit_time[row] += I[col + row * total_states];
    }

    printf("Expected absorption time from each state:\n");
    for (int i = 0; i < total_states; i++) {
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


double mean(const u_int8_t *values, const long n) {
    if (values == NULL || n <= 0) {
        fprintf(stderr, "Error: Invalid input to mean function.\n");
        return NAN;
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += values[i];
    }
    return sum / (double) n;
}

double p_st_dev(const uint8_t *values, const long n, const double avg) {
    if (values == NULL) {
        fprintf(stderr, "Error: Invalid input to p_st_dev function.\n");
        return NAN;
    }
    if (n <= 1) {
        fprintf(stderr, "Error: At least two elements are required to compute standard deviation.\n");
        return NAN;
    }
    double sq_sum = 0.0;
    for (int i = 0; i < n; i++)
        sq_sum += (values[i] - avg) * (values[i] - avg);
    return sqrt(sq_sum / (double) n);
}

void montecarlo_simulation(const long attempts, const u_int8_t max_players) {
    printf("Starting montecarlo simulation...\n");
    double *avg = calloc(max_players, sizeof(*avg));
    if (avg == NULL) exit(EXIT_FAILURE);
    double *st_dev = calloc(max_players, sizeof(*st_dev));
    if (st_dev == NULL) exit(EXIT_FAILURE);
    #pragma omp parallel for num_threads(8) schedule(dynamic) default(none) shared(attempts, max_players, avg, st_dev)
    for (u_int8_t n_players = 1; n_players <= max_players; n_players++) {
        unsigned int seed = (unsigned int) time(NULL) ^ omp_get_thread_num();
        // array to store the number of turns each attempt took to finish the game
        u_int8_t *turns_per_attempt = calloc(attempts, sizeof(*turns_per_attempt));
        if (turns_per_attempt == NULL) exit(EXIT_FAILURE);
        // array to store the position of each player
        u_int8_t *positions = calloc(n_players, sizeof(*positions));
        if (positions == NULL) exit(EXIT_FAILURE);
        for (long int attempt = 0; attempt < attempts; attempt++) {
            int8_t winner = -1;
            memset(positions, 0, n_players * sizeof(*positions));
            long int turns = 1;
            while (1) {
                for (u_int8_t i = 0; i < n_players; i++) {
                    const u_int8_t dice = rand_r(&seed) % 6 + 1;
                    positions[i] += dice;
                    if (positions[i] >= N) {
                        winner = (int8_t) i;
                        break;
                    }
                }
                if (winner != -1) break;
                turns++;
            }
            turns_per_attempt[attempt] = turns;
        }

        const double m = mean(turns_per_attempt, attempts);
        const double s = p_st_dev(turns_per_attempt, attempts, m);
        avg[n_players - 1] = m;
        st_dev[n_players - 1] = s;
        free(turns_per_attempt);
        free(positions);
    }
    printf("\n\nExpected # of turns for increasing number of players:\n");
    for (int i = 0; i < max_players; i++)
        printf("(%d, %.6f)\n", i + 1, avg[i]);

    printf("\nSt_dev:\n");
    for (int i = 0; i < max_players; i++)
        printf("(%d, %.6f)\n", i + 1, st_dev[i]);
    free(avg);
    free(st_dev);
}
