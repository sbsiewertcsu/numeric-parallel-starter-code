#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>

typedef struct {
    unsigned char r, g, b;
} Pixel;

typedef struct {
    int width;
    int height;
    int maxval;
    Pixel *pixels;
} PPMImage;

static void free_ppm(PPMImage *img) {
    if (img) {
        free(img->pixels);
        img->pixels = NULL;
    }
}

//static void fail(const char *msg) {
//    fprintf(stderr, "Error: %s\n", msg);
//    exit(1);
//}

static void skip_comments_and_whitespace(FILE *fp) {
    int ch;
    for (;;) {
        ch = fgetc(fp);
        if (ch == EOF) {
            return;
        }

        if (isspace(ch)) {
            continue;
        }

        if (ch == '#') {
            while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
            }
            continue;
        }

        ungetc(ch, fp);
        return;
    }
}

static int read_token(FILE *fp, char *buf, size_t bufsize) {
    size_t i = 0;
    int ch;

    skip_comments_and_whitespace(fp);

    ch = fgetc(fp);
    if (ch == EOF) {
        return 0;
    }

    while (ch != EOF && !isspace(ch) && ch != '#') {
        if (i + 1 < bufsize) {
            buf[i++] = (char)ch;
        }
        ch = fgetc(fp);
    }

    if (ch == '#') {
        while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
        }
    }

    buf[i] = '\0';
    return 1;
}

/* Simple 64-bit LCG PRNG */
static uint64_t rng_state = 1;

static void rng_seed(uint64_t seed) {
    if (seed == 0) {
        seed = 1;
    }
    rng_state = seed;
}

static uint64_t rng_next_u64(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1ULL;
    return rng_state;
}

static size_t rng_range(size_t upper_exclusive) {
    if (upper_exclusive == 0) {
        return 0;
    }
    return (size_t)(rng_next_u64() % (uint64_t)upper_exclusive);
}

static int read_ppm(const char *filename, PPMImage *img) {
    FILE *fp = NULL;
    char token[64];
    char magic[3];
    size_t count, i;
    int r, g, b;

    img->width = 0;
    img->height = 0;
    img->maxval = 255;
    img->pixels = NULL;

    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: could not open input file: %s\n", filename);
        return 0;
    }

    if (!read_token(fp, token, sizeof(token))) {
        fclose(fp);
        fprintf(stderr, "Error: failed to read PPM magic number\n");
        return 0;
    }

    if (strcmp(token, "P3") != 0 && strcmp(token, "P6") != 0) {
        fclose(fp);
        fprintf(stderr, "Error: unsupported PPM format: %s\n", token);
        return 0;
    }

    strncpy(magic, token, sizeof(magic));
    magic[2] = '\0';

    if (!read_token(fp, token, sizeof(token))) {
        fclose(fp);
        fprintf(stderr, "Error: failed to read width\n");
        return 0;
    }
    img->width = atoi(token);

    if (!read_token(fp, token, sizeof(token))) {
        fclose(fp);
        fprintf(stderr, "Error: failed to read height\n");
        return 0;
    }
    img->height = atoi(token);

    if (!read_token(fp, token, sizeof(token))) {
        fclose(fp);
        fprintf(stderr, "Error: failed to read maxval\n");
        return 0;
    }
    img->maxval = atoi(token);

    if (img->width <= 0 || img->height <= 0) {
        fclose(fp);
        fprintf(stderr, "Error: invalid image dimensions\n");
        return 0;
    }

    if (img->maxval != 255) {
        fclose(fp);
        fprintf(stderr, "Error: only maxval=255 is supported\n");
        return 0;
    }

    count = (size_t)img->width * (size_t)img->height;
    img->pixels = (Pixel *)malloc(count * sizeof(Pixel));
    if (!img->pixels) {
        fclose(fp);
        fprintf(stderr, "Error: out of memory\n");
        return 0;
    }

    if (strcmp(magic, "P3") == 0) {
        for (i = 0; i < count; ++i) {
            if (!read_token(fp, token, sizeof(token))) {
                free_ppm(img);
                fclose(fp);
                fprintf(stderr, "Error: failed reading P3 pixel data\n");
                return 0;
            }
            r = atoi(token);

            if (!read_token(fp, token, sizeof(token))) {
                free_ppm(img);
                fclose(fp);
                fprintf(stderr, "Error: failed reading P3 pixel data\n");
                return 0;
            }
            g = atoi(token);

            if (!read_token(fp, token, sizeof(token))) {
                free_ppm(img);
                fclose(fp);
                fprintf(stderr, "Error: failed reading P3 pixel data\n");
                return 0;
            }
            b = atoi(token);

            if (r < 0 || r > 255 || g < 0 || g > 255 || b < 0 || b > 255) {
                free_ppm(img);
                fclose(fp);
                fprintf(stderr, "Error: P3 pixel value out of range\n");
                return 0;
            }

            img->pixels[i].r = (unsigned char)r;
            img->pixels[i].g = (unsigned char)g;
            img->pixels[i].b = (unsigned char)b;
        }
    } else {
        int ch;

        do {
            ch = fgetc(fp);
        } while (ch != EOF && isspace(ch));

        if (ch == EOF) {
            free_ppm(img);
            fclose(fp);
            fprintf(stderr, "Error: missing P6 pixel data\n");
            return 0;
        }
        ungetc(ch, fp);

        for (i = 0; i < count; ++i) {
            unsigned char rgb[3];
            if (fread(rgb, 1, 3, fp) != 3) {
                free_ppm(img);
                fclose(fp);
                fprintf(stderr, "Error: failed reading P6 pixel data\n");
                return 0;
            }
            img->pixels[i].r = rgb[0];
            img->pixels[i].g = rgb[1];
            img->pixels[i].b = rgb[2];
        }
    }

    fclose(fp);
    return 1;
}

static int write_ppm(const char *filename, const PPMImage *img) {
    FILE *fp = fopen(filename, "wb");
    size_t count, i;

    if (!fp) {
        fprintf(stderr, "Error: could not open output file: %s\n", filename);
        return 0;
    }

    fprintf(fp, "P6\n%d %d\n%d\n", img->width, img->height, img->maxval);

    count = (size_t)img->width * (size_t)img->height;
    for (i = 0; i < count; ++i) {
        unsigned char rgb[3];
        rgb[0] = img->pixels[i].r;
        rgb[1] = img->pixels[i].g;
        rgb[2] = img->pixels[i].b;
        if (fwrite(rgb, 1, 3, fp) != 3) {
            fclose(fp);
            fprintf(stderr, "Error: failed writing output image\n");
            return 0;
        }
    }

    fclose(fp);
    return 1;
}

static size_t *make_permutation(size_t n, uint64_t key) {
    size_t *perm;
    size_t i;

    perm = (size_t *)malloc(n * sizeof(size_t));
    if (!perm) {
        return NULL;
    }

    for (i = 0; i < n; ++i) {
        perm[i] = i;
    }

    rng_seed(key);
    if (n > 1) {
        for (i = n - 1; i > 0; --i) {
            size_t j = rng_range(i + 1);
            size_t tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }
    }

    return perm;
}

static int scramble_image(PPMImage *img, uint64_t key) {
    size_t n = (size_t)img->width * (size_t)img->height;
    size_t *perm = NULL;
    Pixel *out = NULL;
    size_t i;

    perm = make_permutation(n, key);
    if (!perm) {
        fprintf(stderr, "Error: out of memory\n");
        return 0;
    }

    out = (Pixel *)malloc(n * sizeof(Pixel));
    if (!out) {
        free(perm);
        fprintf(stderr, "Error: out of memory\n");
        return 0;
    }

    for (i = 0; i < n; ++i) {
        out[i] = img->pixels[perm[i]];
    }

    free(img->pixels);
    img->pixels = out;
    free(perm);
    return 1;
}

int main(int argc, char *argv[]) {
    PPMImage img;
    uint64_t key;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.ppm> <output.ppm> <key>\n", argv[0]);
        return 1;
    }

    key = (uint64_t)strtoull(argv[3], NULL, 10);

    if (!read_ppm(argv[1], &img)) {
        return 1;
    }

    if (!scramble_image(&img, key)) {
        free_ppm(&img);
        return 1;
    }

    if (!write_ppm(argv[2], &img)) {
        free_ppm(&img);
        return 1;
    }

    free_ppm(&img);
    printf("Scrambled image written to %s\n", argv[2]);
    return 0;
}
