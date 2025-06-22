import struct
import tiktoken

with open("tiny.txt") as f:
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(f.read())

with open("data", "wb") as f_d:
    for i in tokens:
        v = struct.pack("H", i)
        f_d.write(v)

structs = []
words = b""

# get vocab table
for i in range(50257):
    # word = enc.decode([i])
    word = enc.decode_single_token_bytes(i)
    offset = len(words)
    structs.append(offset)
    size = len(word)
    structs.append(size)
    # words += bytes(word, "utf-8")
    words += word

with open("enc", "wb") as f_out:
    for st in structs:
        v = struct.pack("I", st)
        f_out.write(v)
    f_out.write(words)
print(len(words))



