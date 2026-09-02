#pragma once

#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ini.h"

static void puf_json_write_key(FILE* fp, const char* key) {
    fputc('"', fp);
    for (const unsigned char* p = (const unsigned char*)key; *p; p++) {
        switch (*p) {
        case '"':
            fputs("\\\"", fp);
            break;
        case '\\':
            fputs("\\\\", fp);
            break;
        case '\b':
            fputs("\\b", fp);
            break;
        case '\f':
            fputs("\\f", fp);
            break;
        case '\n':
            fputs("\\n", fp);
            break;
        case '\r':
            fputs("\\r", fp);
            break;
        case '\t':
            fputs("\\t", fp);
            break;
        default:
            if (*p < 0x20) {
                fprintf(fp, "\\u%04x", *p);
            } else {
                fputc(*p, fp);
            }
        }
    }
    fputc('"', fp);
}

static void puf_live_metrics_write(const char* metrics_dir, long epoch,
        Dict* log) {
    if (!metrics_dir || strcmp(metrics_dir, "None") == 0) {
        return;
    }

    char path[4096];
    char tmp[4096];
    snprintf(path, sizeof(path), "%s/epoch-%06ld.jsonl", metrics_dir, epoch);
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, getpid());
    int fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL, 0666);
    assert(fd >= 0 && "failed to create live metric chunk");
    FILE* fp = fdopen(fd, "w");
    assert(fp && "failed to open live metric chunk");

    struct timespec ts;
    assert(clock_gettime(CLOCK_REALTIME, &ts) == 0
        && "failed to read live metric timestamp");
    double timestamp = (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
    fprintf(fp, "{\"_step\":%ld,\"_timestamp\":%.9f", epoch, timestamp);
    for (int i = 0; i < log->size; i++) {
        DictItem* item = &log->items[i];
        if (item->str || item->values || !isfinite(item->value)
                || strcmp(item->key, "_step") == 0
                || strcmp(item->key, "_timestamp") == 0) {
            continue;
        }
        fputc(',', fp);
        puf_json_write_key(fp, item->key);
        fprintf(fp, ":%.17g", item->value);
    }
    fputs("}\n", fp);
    assert(fflush(fp) == 0 && "failed to flush live metric chunk");
    assert(fsync(fd) == 0 && "failed to sync live metric chunk");
    assert(fclose(fp) == 0 && "failed to close live metric chunk");
    assert(rename(tmp, path) == 0 && "failed to publish live metric chunk");
}
