#pragma once

#define PUF_ENV_DECODER_HEADER_V1 1

#ifdef ENV_DECODER_HEADER
#include ENV_DECODER_HEADER
#endif

static void create_custom_decoder(const char* env_name, Decoder* dec) {
#ifdef ENV_DECODER_HEADER
    create_env_decoder(dec);
    return;
#endif
#ifdef PUFFER_NETHACK
    if (strcmp(env_name, "nethack") == 0) {
        create_nethack_decoder(dec);
        return;
    }
#endif
}
