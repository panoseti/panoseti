# Functions for reading/writing/modifying quabo config files.
# These are structured as KEY=VALUE lines;
# everything after '*' is ignored
#
# We parse them into a list of [key, value] pairs rather than a dictionary,
# in order to preserve the order of lines

def read(path):
    with open(path) as f:
        c = []
        for line in f.readlines():
            n = line.find('*')
            if n >= 0:
                line = line[0:n]
            n = line.find('=')
            if n < 0:
                continue
            key = line[0:n].strip()
            value = line[n+1:].strip()
            c.append([key, value])
    return c

def write(c, path):
    with open(path, 'w') as f:
        for x in c:
            f.write('%s=%s\n'%(x[0], x[1]))

def get(c, key):
    for x in c:
        if x[0] == key:
            return x[1]
    raise Exception('key not found')

def set(c, key, value):
    for x in c:
        if x[0] == key:
            x[1] = value
            return
    raise Exception('key not found')

def test():
    c = read('quabo_config.txt')
    set(c, 'SMALL_DAC', 'foo')
    print(get(c, 'SMALL_DAC'))
    write(c, 'foo.txt')

# test()
