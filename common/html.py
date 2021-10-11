from urllib.parse import quote, unquote


def encode_url(str):
    return quote(str, encoding='utf-8')


def decode_url(str):
    return unquote(str)


if __name__ == '__main__':
    s = "我是 Flash, 大家好"
    s1 = encode_url(s)
    s2 = decode_url(s1)
    print(f"s1='{s1}'")
    print(f"s1='{s2}'")
