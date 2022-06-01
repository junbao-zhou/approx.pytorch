one_thread_compute = (8, 4, 4)

thread_block = (16, 8, 8)

weight_shared_size = (
    one_thread_compute[0] + thread_block[0] - 1, ) + one_thread_compute[1:]

output_shared_size = thread_block

stride = (1, 1)


def input_size(kernel_size, stride, steps):
    return (steps - 1) * stride + kernel_size


input_shared_size = (
    weight_shared_size[0],
    input_size(one_thread_compute[1], stride[0], thread_block[1]),
    input_size(one_thread_compute[2], stride[1], thread_block[2]),
)


def elem_num(tup):
    return tup[0] * tup[1] * tup[2]


total_mem = elem_num(weight_shared_size) + \
    elem_num(output_shared_size) + elem_num(input_shared_size)

print(f"""
weight_shared_size = {weight_shared_size}
output_shared_size = {output_shared_size}
input_shared_size = {input_shared_size}
total_mem = {total_mem}
""")
