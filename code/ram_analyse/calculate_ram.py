def calculate(M, N, B, F):

    all = M * B * F + M * N * F * 2 + M * B * F + N * B * F + M * F + N * F
    all = all / (1024 ** 3)
    print("一共{all}GB".format(all=all))
    print("")


def shuffle_ram(context_size, n_batches_in_buffer, d_in, store_batch_size_prompts, num_layer,F):
    half_buffer_size = n_batches_in_buffer // 2

    total_size = store_batch_size_prompts * half_buffer_size
    buffer_size = total_size * context_size

    half_store_batch_size_prompts=store_batch_size_prompts//2

    buffer_count = 19
    all = buffer_count* buffer_size * num_layer * d_in * F

    all = all / (1024 ** 3)
    print("一共{all}GB".format(all=all))


if __name__ == '__main__':
    # for M in [512,768,1024,4096]:
    #     for N in [2**12,2**15,2**17,2**20,2**22,2**25]:
    #         ef=N/M
    #     # for ef in [2**3]:
    #     #     N=M*ef
    #         print("M={M},N={N}的时候,ef={ef}".format(M=M,N=N,ef=ef),end="")
    #         calculate(M,N,4096,4)
    shuffle_ram(512, 64, 4096, 16, 1, 4)
    for context_size in [128,256,512]:
        for n_batches_in_buffer in [16,32,64]:
            for store_batch_size_prompts in [4,8,16]:
                print("context_size={a},n_batches_in_buffer={b},store_batch_size_prompts={c}的时候".format(a=context_size, b=n_batches_in_buffer,c=store_batch_size_prompts), end="")
                shuffle_ram(context_size, n_batches_in_buffer, 4096, store_batch_size_prompts, 1, 1)
