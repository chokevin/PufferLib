#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

static char* puf_read_checkpoint_exact(const char* path, int64_t nbytes) {
    assert(nbytes >= 0 && "invalid weights size");
    FILE* fp = fopen(path, "rb");
    assert(fp && "failed to open weights for reading");

    struct stat info;
    assert(fstat(fileno(fp), &info) == 0 && "failed to stat weights");
    assert(info.st_size == nbytes && "weights file size mismatch");

    size_t expected = (size_t)nbytes;
    assert((int64_t)expected == nbytes && "weights size exceeds address space");
    char* buf = (char*)malloc(expected ? expected : 1);
    assert(buf && "failed to allocate weights buffer");
    size_t nread = fread(buf, 1, expected, fp);
    assert(nread == expected && !ferror(fp) && "failed to read weights");
    assert(fclose(fp) == 0 && "failed to close weights after reading");
    return buf;
}
