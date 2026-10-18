
    // --- IF-cache weight-size sweep v2 (m fixed, k varies) ---
    for (int bs : {1, 16}) {
        for (int64_t k : { 1024, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 20480, 24576, 32768, 49152, 65536 }) {
            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 4096, bs, k, {1, 1}, {1, 1}));
        }
        for (int64_t k : { 2048, 3840, 7680, 11520, 15360, 19200, 23040, 26880, 30720, 38656, 46336, 61696, 92416, 123392 }) {
            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 4096, bs, k, {1, 1}, {1, 1}));
        }
        for (int64_t k : { 3584, 7168, 14592, 21760, 29184, 36352, 43776, 50944, 58368, 72704, 87296, 116480, 174848, 232960 }) {
            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 4096, bs, k, {1, 1}, {1, 1}));
        }
        for (int64_t k : { 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 16384, 24576, 32768 }) {
            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 4096, bs, k, {1, 1}, {1, 1}));
        }
    }
