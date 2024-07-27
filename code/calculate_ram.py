def calculate(M, N, B, F):
    all = M * B * F + M * N * F * 2 + M * B * F + N * B * F + M * F + N * F
    all = all / (1024 ** 3)
    print("一共{all}GB".format(all=all))


def shuffle_ram(context_size, n_batches_in_buffer, d_in, store_batch_size_prompts, num_layer,
                F):
    half_buffer_size = n_batches_in_buffer // 2
    # total_training_tokens = total_training_steps * batch_size
    total_size = store_batch_size_prompts * half_buffer_size
    buffer_size = total_size * context_size
    # buffer_count = batch_size / total_size
    # buffer_count = half_buffer_size+(half_buffer_size-1)*2+half_buffer_size+1+(half_buffer_size-2)*2
    half_store_batch_size_prompts=store_batch_size_prompts//2
    # buffer_count = half_store_batch_size_prompts+(half_store_batch_size_prompts-1)*2+half_store_batch_size_prompts+1+(half_store_batch_size_prompts-2)*2
    buffer_count = 9+5*2
    # buffer_count=1
    # all = buffer_count * buffer_size * num_layer * d_in * 4
    # all = batch_size * context_size * num_layer * d_in * 4 *store_batch_size_prompts
    # all = n_batches_in_buffer * buffer_size * num_layer * d_in * 4
    all = buffer_count* buffer_size * num_layer * d_in * F
    # all = buffer_size * num_layer * d_in * 2
    all = all / (1024 ** 3)
    print("一共{all}GB".format(all=all))


if __name__ == '__main__':
    # for M in [4096]:
    #     for N in [2**3]:
    #         ef=N/M
    #     # for ef in [2**4,2**5]:
    #     #     N=M*ef
    #         print("M={M},N={N}的时候,ef={ef}".format(M=M,N=N,ef=ef),end="")
    #         calculate(M,N,1024,4)
    shuffle_ram(512, 64, 4096, 16, 1, 4)
    # for context_size in [128,256,512]:
    #     for n_batches_in_buffer in [16,32,64]:
    #         for store_batch_size_prompts in [4,8,16]:
    #             print("context_size={a},n_batches_in_buffer={b},store_batch_size_prompts={c}的时候".format(a=context_size, b=n_batches_in_buffer,c=store_batch_size_prompts), end="")
    #             shuffle_ram(context_size, n_batches_in_buffer, 4096, store_batch_size_prompts, 4096, 1, 30_000)
